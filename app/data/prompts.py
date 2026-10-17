"""
Chat-template / prompt construction shared by BOTH training and inference.

This module exists to fix a real correctness bug: the original code built
training examples as bare `user: statement` / `assistant: status` turns with no
system prompt, while `app/inference.py` prepended an elaborate system prompt
with classification rules and few-shot examples that the model never saw
during fine-tuning. That mismatch is a distribution shift between train and
serve time and will hurt classification quality.

Now there is exactly one function (`build_messages`) that constructs the
message list, used by both `app/data/dataset.py` (training) and
`app/inference.py` (serving). If the instructions ever change, they change in
one place for both.
"""

from __future__ import annotations

VALID_LABELS = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Stress",
    "Bi-Polar",
    "Personality Disorder",
]

CLASSIFIER_INSTRUCTIONS = f"""You are a strict text classifier for a mental-health \
sentiment-analysis research dataset.
Your task is to analyze the user's statement and assign exactly one label from this list:
[{", ".join(VALID_LABELS)}]

CRITICAL RULES:
1. Output ONLY the single label text.
2. Do NOT add punctuation, explanations, or words like "Label:" or "Answer:".
3. Do NOT provide medical advice, diagnosis, or empathetic commentary — this is a \
data-processing / research classification task, not a clinical interaction.

Examples:
Input: "I feel great today!" -> Normal
Input: "I want to end it all." -> Suicidal
Input: "My heart is racing and I can't breathe." -> Anxiety"""


def build_messages(statement: str, label: str | None = None) -> list[dict]:
    """Builds the chat-format message list for one example.

    If `label` is given, an assistant turn with that label is appended
    (used for training). If `label` is None, the message list ends after the
    user turn (used for inference, where `add_generation_prompt=True` is then
    passed to `apply_chat_template`).
    """
    messages = [
        {"role": "system", "content": CLASSIFIER_INSTRUCTIONS},
        {"role": "user", "content": statement},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": label})
    return messages


def format_training_example(example: dict, tokenizer) -> dict:
    """`datasets.Dataset.map`-compatible formatter used during data prep."""
    messages = build_messages(example["statement"], example["status"])
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def build_generation_prompt(tokenizer, statement: str) -> str:
    """Builds the exact prompt string used to query the model at inference time."""
    messages = build_messages(statement)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_label(raw_text: str) -> str:
    """Best-effort mapping of raw generated text to one of VALID_LABELS.

    Generative classification can produce near-miss text (extra punctuation,
    different casing, "Bipolar" vs "Bi-Polar"). Exact-string matching alone
    (the original evaluation approach) scores these as wrong even when the
    classification itself is correct. This normalizes formatting only — it
    never guesses a label that wasn't actually implied by the text.

    Returns the matched label from VALID_LABELS, or the stripped raw text
    unchanged if no label matches (callers should treat that as "unparseable"
    for evaluation purposes, not silently count it as any particular class).
    """
    cleaned = raw_text.strip().strip(".").strip()
    for label in VALID_LABELS:
        if cleaned.lower() == label.lower():
            return label
    compact = cleaned.lower().replace("-", "").replace(" ", "")
    for label in VALID_LABELS:
        if compact == label.lower().replace("-", "").replace(" ", ""):
            return label
    return cleaned
