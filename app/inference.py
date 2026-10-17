"""
Inference entrypoint. Supports:

- `--mode eval`: run over the (stratified) test split and report real
  classification metrics (accuracy, macro/weighted F1, per-class P/R/F1,
  confusion matrix) — not just raw accuracy, since the dataset is imbalanced.
- `--mode chat`: interactive single-statement classification.

IMPORTANT: this is an experimental text-classification research model, not a
clinical or diagnostic tool. See README "Safety Considerations".

Run with:
    poetry run python -m app.inference --mode eval --num-samples 50
    poetry run python -m app.inference --mode chat
"""

from __future__ import annotations

# Unsloth must be imported before transformers/peft/trl for its optimizations
# to patch cleanly — see the matching comment in app/train.py for why this
# has to happen here rather than inside app/model/model.py's Unsloth branch.
import unsloth  # noqa: F401,E402

import argparse
import logging

import torch
from transformers import TextStreamer

from app.configs.settings import get_settings
from app.data.dataset import prepare_dataset
from app.data.prompts import build_generation_prompt, normalize_label
from app.evaluation.metrics import compute_metrics
from app.model.model import load_model
from app.model.tokenizer import load_tokenizer
from app.utils.environment import log_environment_summary, require_cuda
from app.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

SAFETY_NOTICE = (
    "⚠️  This model is an experimental research text-classifier trained on a "
    "public sentiment-analysis dataset. It is NOT a diagnostic or clinical tool "
    "and must not be used to make decisions about a person's mental health. If "
    "you or someone you know is struggling, please contact a mental health "
    "professional or a crisis line."
)


def generate_response(model, tokenizer, input_text: str, use_streamer: bool = False) -> str:
    """Classifies one statement using the exact same prompt format the model
    was fine-tuned on (see app/data/prompts.py)."""
    prompt = build_generation_prompt(tokenizer, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if use_streamer else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # a label is a few tokens; 2048 was excessive and slowed inference
            temperature=1.0,
            top_p=0.95,
            top_k=64,  # recommended Gemma-3 generation settings
            streamer=streamer,
        )

    prompt_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def run_evaluation(model, tokenizer, settings, num_samples: int | None = None):
    """Runs generation over the test split and reports real classification metrics."""
    logger.info("Starting evaluation on test data...")

    _, test_data, _ = prepare_dataset(
        dataset_path=settings.require_dataset_path(),
        tokenizer=tokenizer,
        test_size=settings.test_size,
        seed=settings.seed,
    )

    if num_samples is not None:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    logger.info("Processing %d examples...", len(test_data))

    y_true, y_pred = [], []
    for row in test_data:
        statement = row["statement"]
        actual = row["status"]

        raw_prediction = generate_response(model, tokenizer, statement)
        prediction = normalize_label(raw_prediction)

        y_true.append(actual)
        y_pred.append(prediction)

        marker = "✅" if prediction == actual else "❌"
        logger.info("True: %-22s | Pred: %-22s | %s", actual, prediction, marker)

    result = compute_metrics(y_true, y_pred, split="test")
    print("\n" + result.human_summary())

    metrics_path = f"{settings.run_output_dir}/eval_metrics.json"
    result.save(metrics_path)
    logger.info("Saved machine-readable metrics to %s", metrics_path)
    return result


def run_demo(model, tokenizer) -> None:
    """Interactive classification loop. Handles empty/invalid input gracefully."""
    print(SAFETY_NOTICE)
    print("\n--- Interactive Demo (type 'q' to quit) ---")

    while True:
        try:
            user_input = input("\nStatement: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.strip().lower() == "q":
            break
        if not user_input.strip():
            print("Please enter some text (or 'q' to quit).")
            continue

        print("Prediction:", end=" ")
        raw = generate_response(model, tokenizer, user_input, use_streamer=True)
        label = normalize_label(raw)
        if label != raw.strip():
            print(f"\n(normalized to: {label})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference/evaluation for the fine-tuned classifier.")
    parser.add_argument("--mode", choices=["eval", "chat"], required=True)
    parser.add_argument("--num-samples", type=int, default=None, help="Limit eval to N test examples.")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--lora-mode", choices=["lora", "qlora"], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()

    overrides = {}
    if args.dataset_path is not None:
        overrides["dataset_path"] = args.dataset_path
    if args.lora_mode is not None:
        overrides["lora_mode"] = args.lora_mode

    settings = get_settings(**overrides)

    require_cuda()  # fail fast with an actionable message rather than a deep bitsandbytes crash
    log_environment_summary()

    model = load_model(settings, inference=True)
    tokenizer = load_tokenizer(settings.model_id, inference=True)

    if args.mode == "eval":
        run_evaluation(model, tokenizer, settings, num_samples=args.num_samples)
    else:
        run_demo(model, tokenizer)
