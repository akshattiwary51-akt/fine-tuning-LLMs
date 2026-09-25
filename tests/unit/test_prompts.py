import pytest

from app.data.prompts import (
    VALID_LABELS,
    build_generation_prompt,
    build_messages,
    format_training_example,
    normalize_label,
)


def test_build_messages_training_includes_assistant_turn():
    messages = build_messages("I feel great", "Normal")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant"]
    assert messages[-1]["content"] == "Normal"


def test_build_messages_inference_excludes_assistant_turn():
    messages = build_messages("I feel great")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]


def test_train_and_inference_share_identical_system_prompt():
    """Regression test for the original train/inference prompt-mismatch bug:
    training built bare user/assistant turns with no system prompt, while
    inference used an elaborate instruction+few-shot system prompt the model
    never saw during fine-tuning."""
    train_messages = build_messages("some statement", "Normal")
    inference_messages = build_messages("some statement")
    assert train_messages[0]["role"] == "system"
    assert inference_messages[0]["role"] == "system"
    assert train_messages[0]["content"] == inference_messages[0]["content"]


def test_format_training_example_uses_tokenizer_chat_template(fake_tokenizer):
    example = {"statement": "hello", "status": "Normal"}
    result = format_training_example(example, fake_tokenizer)
    assert "text" in result
    assert "system:" in result["text"]
    assert "assistant:Normal" in result["text"]


def test_build_generation_prompt_adds_generation_marker(fake_tokenizer):
    prompt = build_generation_prompt(fake_tokenizer, "hello")
    assert prompt.endswith("assistant:")
    assert "assistant:Normal" not in prompt  # no ground-truth label leaked into the prompt


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Normal", "Normal"),
        ("normal", "Normal"),
        ("Normal.", "Normal"),
        ("  Suicidal  ", "Suicidal"),
        ("bipolar", "Bi-Polar"),
        ("bi polar", "Bi-Polar"),
        ("BI-POLAR", "Bi-Polar"),
        ("personality disorder", "Personality Disorder"),
    ],
)
def test_normalize_label_matches_known_labels(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_label_returns_raw_text_when_unrecognized():
    result = normalize_label("this is not a real label at all")
    assert result not in VALID_LABELS
    assert result == "this is not a real label at all"
