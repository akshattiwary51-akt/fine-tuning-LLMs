import csv
from pathlib import Path

import pytest

from app.data.prompts import VALID_LABELS


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's chat-template API, so dataset/prompt
    tests run on CPU without downloading any real model or tokenizer."""

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False):
        rendered = " | ".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += " | assistant:"
        return rendered


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> str:
    """Writes a small, deliberately messy CSV: valid rows for every label, one
    missing statement, one exact duplicate, one invalid label — so cleaning
    logic has something real to remove."""
    path = tmp_path / "synthetic.csv"
    rows = []
    for i in range(70):
        label = VALID_LABELS[i % len(VALID_LABELS)]
        rows.append({"statement": f"Sample statement {i} about {label.lower()}", "status": label})

    rows.append(dict(rows[0]))  # exact duplicate
    rows.append({"statement": "", "status": "Normal"})  # empty statement
    rows.append({"statement": "weird row", "status": "NotARealLabel"})  # invalid label

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["statement", "status"])
        writer.writeheader()
        writer.writerows(rows)

    return str(path)
