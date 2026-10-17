"""
### Sequence length note

Per-sequence, each individual training example (prompt + completion) is
truncated/limited by `Settings.max_seq_length` (default 1024, configurable via
MAX_SEQ_LENGTH). This is the SAME value used both when loading the model
(Unsloth's `max_seq_length`) and in `SFTConfig.max_length` — previously these
were two different hardcoded numbers (2048 vs 1024) that could silently drift
apart. A batch may contain multiple sequences; their combined token count is
not itself limited, only each individual sequence is.

Run with:
    poetry run python -m app.train
    poetry run python -m app.train --lora-mode lora --no-unsloth
    poetry run python -m app.train --dataset-path /path/to/data.csv --max-steps 500
"""

from __future__ import annotations

# Unsloth must be imported before transformers/peft/trl (directly or via
# app.data.dataset / app.model.model / trl below) so its optimizations patch
# cleanly. This has to happen here, at the very top of the entrypoint, not
# inside app/model/model.py's Unsloth branch — by the time that code runs,
# transformers/peft/trl are already loaded via the imports below, which is
# exactly the ordering Unsloth warns against. Harmless to import even when
# USE_UNSLOTH=False for a given run: it's already a required dependency
# either way (see pyproject.toml), this just fixes import order.
import unsloth  # noqa: F401,E402

import argparse
import gc
import json
import logging
import os
import platform
from pathlib import Path
from timeit import default_timer as timer

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from app.configs.settings import Settings, get_settings
from app.data.dataset import prepare_dataset
from app.model.model import count_trainable_parameters, load_model
from app.model.tokenizer import load_tokenizer
from app.utils.environment import (
    gpu_memory_stats,
    log_environment_summary,
    require_cuda,
    reset_peak_memory_stats,
    supports_bf16,
)
from app.utils.logging_utils import setup_logging
from app.utils.seeding import set_seed

logger = logging.getLogger(__name__)

# Ephemeral notebook environments (Colab, Kaggle) can disconnect and wipe all
# local storage mid-run with no warning — this has actually happened during
# real use of this project. Set AUTO_UPLOAD_TO_HUB=true (and configure
# HF_TOKEN + HF_USERNAME/HF_PERSONAL_REPO_ID) to push adapters to Hugging
# Face Hub automatically: once after training completes, AND at every
# checkpoint save during training, so a disconnect only costs progress since
# the last save_steps interval rather than the entire run. Off by default
# since it's a real side effect (uploading to your Hub account) that
# shouldn't happen without explicit opt-in.
AUTO_UPLOAD_TO_HUB = os.environ.get("AUTO_UPLOAD_TO_HUB", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _upload_to_hub(settings: Settings, local_dir: str, context: str) -> None:
    """Best-effort upload; never raises. A failed upload should not crash or
    lose a training run that otherwise succeeded — it just gets logged."""
    try:
        from app.utils.upload import upload_model

        upload_model(
            hf_token=settings.require_hf_token(),
            hf_personal_repo_id=settings.hf_personal_repo_id,
            local_model_dir=local_dir,
        )
        logger.info("Hub upload (%s) succeeded: %s", context, local_dir)
    except Exception:
        logger.exception("Hub upload (%s) failed for %s; continuing.", context, local_dir)


class HubUploadCallback(TrainerCallback):
    """Uploads each checkpoint to Hugging Face Hub as it's saved during
    training, not just once at the end — see AUTO_UPLOAD_TO_HUB above."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        _upload_to_hub(self.settings, checkpoint_dir, context=f"checkpoint step {state.global_step}")


def clear_gpu_memory() -> None:
    """Forces the GPU to release 'reserved' memory back to the system."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("GPU memory cleared.")


def _package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes"):
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = "not installed"
    return versions


def save_run_metadata(settings: Settings, dataset_stats, output_dir: str) -> None:
    """Saves the effective configuration, dataset stats and package versions
    alongside training outputs, so a run can be understood/reproduced later
    without re-reading source code or trusting memory."""
    metadata = {
        "settings": {
            "use_unsloth": settings.use_unsloth,
            "lora_mode": settings.lora_mode,
            "model_id": settings.model_id,
            "max_seq_length": settings.max_seq_length,
            "test_size": settings.test_size,
            "seed": settings.seed,
            "run_name": settings.run_name,
        },
        "dataset_stats": {
            "total_rows": dataset_stats.total_rows,
            "dropped_missing": dataset_stats.dropped_missing,
            "dropped_duplicates": dataset_stats.dropped_duplicates,
            "dropped_invalid_label": dataset_stats.dropped_invalid_label,
            "class_distribution": dataset_stats.class_distribution,
        },
        "package_versions": _package_versions(),
    }
    path = Path(output_dir) / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved run metadata to %s", path)


def save_training_stats(param_stats: dict, memory_stats: dict, train_metrics: dict, output_dir: str) -> None:
    """Saves trainable-parameter counts, GPU memory usage, and the trainer's
    own runtime/throughput metrics (samples/sec, tokens/sec) alongside a
    completed run. This is the actual measured evidence behind any resource-
    efficiency claim — Phase 11 explicitly asks for these numbers to be
    captured rather than asserted."""
    stats = {
        "parameters": param_stats,
        "gpu_memory": memory_stats,
        "trainer_metrics": train_metrics,
    }
    path = Path(output_dir) / "training_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, default=str))
    logger.info("Saved training stats to %s\n%s", path, json.dumps(stats, indent=2, default=str))


def train_model(
    settings: Settings,
    max_steps: int | None = None,
    resume_from_checkpoint: str | bool | None = None,
) -> None:
    """Configures and runs the SFT training pipeline for the given settings.

    Args:
        resume_from_checkpoint: None (default) starts fresh. True auto-detects
            the latest checkpoint under settings.run_output_dir (via
            transformers' get_last_checkpoint). A string resumes from that
            exact checkpoint path. Note this only helps when the local
            checkpoint directory still exists — in an ephemeral environment
            where the whole filesystem was wiped (see AUTO_UPLOAD_TO_HUB
            above), there is no local checkpoint left to resume from; you'd
            need to download the last uploaded checkpoint from Hub first.
    """
    require_cuda()  # fail fast with an actionable message rather than a deep bitsandbytes crash
    log_environment_summary()

    if AUTO_UPLOAD_TO_HUB:
        # Validate credentials/destination now, not at the first checkpoint
        # 10+ minutes into a run — a misconfiguration should fail immediately.
        settings.require_hf_token()
        _ = settings.hf_personal_repo_id
        logger.info(
            "AUTO_UPLOAD_TO_HUB is enabled: adapters will be pushed to %s "
            "at every checkpoint and after training.",
            settings.hf_personal_repo_id,
        )
    else:
        logger.info(
            "AUTO_UPLOAD_TO_HUB is disabled (default) — adapters stay local only. "
            "In an ephemeral environment (Colab, Kaggle), set AUTO_UPLOAD_TO_HUB=true "
            "to avoid losing progress on a disconnect."
        )

    logger.info(
        "Starting training | backend=%s | lora_mode=%s | model=%s",
        settings.backend_name,
        settings.lora_mode,
        settings.model_id,
    )

    set_seed(settings.seed)
    clear_gpu_memory()

    model = load_model(settings, inference=False)
    tokenizer = load_tokenizer(settings.model_id, inference=False)

    param_stats = count_trainable_parameters(model)
    logger.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        f"{param_stats['trainable_params']:,}",
        f"{param_stats['total_params']:,}",
        param_stats["trainable_pct"],
    )

    dataset_path = settings.require_dataset_path()
    train_data, test_data, dataset_stats = prepare_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        test_size=settings.test_size,
        seed=settings.seed,
    )

    save_run_metadata(settings, dataset_stats, settings.run_output_dir)

    # bfloat16 needs Ampere+ (A100, RTX 30xx/40xx, L4, ...); older GPUs —
    # notably Colab's free-tier Tesla T4 (Turing) — only support fp16.
    # `TrainingArguments(bf16=True)` raises a hard error rather than falling
    # back automatically, so this has to be checked explicitly.
    use_bf16 = supports_bf16()
    logger.info("Mixed precision: %s", "bf16" if use_bf16 else "fp16 (no bf16 support on this GPU)")

    training_args = SFTConfig(
        dataset_text_field="text",
        output_dir=settings.run_output_dir,
        max_steps=max_steps if max_steps is not None else 1200,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        max_length=settings.max_seq_length,  # single source of truth, see module docstring
        warmup_steps=200,
        seed=settings.seed,
        # Precision & memory optimization
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        fp16=not use_bf16,
        fp16_full_eval=not use_bf16,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_num_proc=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=600,
        save_steps=600,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Never prompt interactively — this would silently hang any
        # unattended run (Colab, a background job, CI). Explicit opt-in only:
        # set REPORT_TO (e.g. "wandb", "tensorboard") if you want an
        # experiment tracker; defaults to none.
        report_to=os.environ.get("REPORT_TO", "none"),
        # Throughput reporting (Phase 11: measure, don't just claim, efficiency).
        # Requires transformers>=4.46 (this project's floor already meets that).
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        processing_class=tokenizer,
        callbacks=[HubUploadCallback(settings)] if AUTO_UPLOAD_TO_HUB else None,
    )

    processed_sample = trainer.train_dataset[0]
    logger.info(
        "First processed sample length: %d (max_length=%d)",
        len(processed_sample["input_ids"]),
        training_args.max_length,
    )

    resume_checkpoint: str | None = None
    if resume_from_checkpoint:
        if isinstance(resume_from_checkpoint, str):
            resume_checkpoint = resume_from_checkpoint
            logger.info("Resuming from explicit checkpoint: %s", resume_checkpoint)
        else:
            resume_checkpoint = get_last_checkpoint(settings.run_output_dir)
            if resume_checkpoint:
                logger.info("Auto-detected checkpoint to resume from: %s", resume_checkpoint)
            else:
                logger.warning(
                    "--resume was set but no checkpoint found in %s; starting fresh.",
                    settings.run_output_dir,
                )

    start_time = timer()
    reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    elapsed_minutes = (timer() - start_time) / 60
    logger.info("Training complete. Time: %.2f minutes", elapsed_minutes)
    logger.info("Trainer metrics: %s", train_result.metrics)

    memory_stats = gpu_memory_stats()
    logger.info(
        "Peak GPU memory: %.2f GB allocated / %.2f GB reserved",
        memory_stats["max_allocated_gb"],
        memory_stats["max_reserved_gb"],
    )
    save_training_stats(param_stats, memory_stats, train_result.metrics, settings.run_output_dir)

    logger.info("Saving adapters to: %s", settings.adapter_dir)
    trainer.model.save_pretrained(settings.adapter_dir)
    trainer.processing_class.save_pretrained(settings.adapter_dir)

    if AUTO_UPLOAD_TO_HUB:
        _upload_to_hub(settings, settings.adapter_dir, context="final adapters")

    del model
    del trainer
    clear_gpu_memory()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-1B-IT with LoRA/QLoRA.")
    parser.add_argument("--lora-mode", choices=["lora", "qlora"], default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="CHECKPOINT_DIR",
        help=(
            "Resume training. Bare '--resume' auto-detects the latest checkpoint "
            "under this run's output directory; '--resume /path/to/checkpoint-600' "
            "resumes from an explicit path. Omit to start fresh (default)."
        ),
    )
    unsloth_group = parser.add_mutually_exclusive_group()
    unsloth_group.add_argument("--unsloth", dest="use_unsloth", action="store_true", default=None)
    unsloth_group.add_argument("--no-unsloth", dest="use_unsloth", action="store_false", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()

    overrides = {}
    if args.lora_mode is not None:
        overrides["lora_mode"] = args.lora_mode
    if args.dataset_path is not None:
        overrides["dataset_path"] = args.dataset_path
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.use_unsloth is not None:
        overrides["use_unsloth"] = args.use_unsloth

    settings = get_settings(**overrides)
    train_model(settings, max_steps=args.max_steps, resume_from_checkpoint=args.resume)
