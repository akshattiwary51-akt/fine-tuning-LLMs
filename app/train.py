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
from trl import SFTConfig, SFTTrainer

from app.configs.settings import Settings, get_settings
from app.data.dataset import prepare_dataset
from app.model.model import load_model
from app.model.tokenizer import load_tokenizer
from app.utils.logging_utils import setup_logging
from app.utils.seeding import set_seed

logger = logging.getLogger(__name__)


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


def train_model(settings: Settings, max_steps: int | None = None) -> None:
    """Configures and runs the SFT training pipeline for the given settings."""
    logger.info(
        "Starting training | backend=%s | lora_mode=%s | model=%s",
        settings.backend_name, settings.lora_mode, settings.model_id,
    )

    set_seed(settings.seed)
    clear_gpu_memory()

    model = load_model(settings, inference=False)
    tokenizer = load_tokenizer(settings.model_id, inference=False)

    dataset_path = settings.require_dataset_path()
    train_data, test_data, dataset_stats = prepare_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        test_size=settings.test_size,
        seed=settings.seed,
    )

    save_run_metadata(settings, dataset_stats, settings.run_output_dir)

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
        bf16=True,
        bf16_full_eval=True,
        fp16=False,
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
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        processing_class=tokenizer,
    )

    processed_sample = trainer.train_dataset[0]
    logger.info(
        "First processed sample length: %d (max_length=%d)",
        len(processed_sample["input_ids"]), training_args.max_length,
    )

    start_time = timer()
    trainer.train()
    elapsed_minutes = (timer() - start_time) / 60
    logger.info("Training complete. Time: %.2f minutes", elapsed_minutes)

    logger.info("Saving adapters to: %s", settings.adapter_dir)
    trainer.model.save_pretrained(settings.adapter_dir)
    trainer.processing_class.save_pretrained(settings.adapter_dir)

    del model
    del trainer
    clear_gpu_memory()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-1B-IT with LoRA/QLoRA.")
    parser.add_argument("--lora-mode", choices=["lora", "qlora"], default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    train_model(settings, max_steps=args.max_steps)
