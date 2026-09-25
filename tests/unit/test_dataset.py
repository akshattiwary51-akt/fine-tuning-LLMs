from collections import Counter

import pytest

from app.data.dataset import load_and_clean, normalize_text, prepare_dataset
from app.data.prompts import VALID_LABELS


def test_normalize_text_strips_newlines_and_tabs():
    assert normalize_text("hello\nworld\t!") == "hello world !"


def test_normalize_text_handles_non_string_input():
    assert normalize_text(None) == ""
    assert normalize_text(123) == ""


def test_load_and_clean_drops_invalid_rows(synthetic_csv):
    df, stats = load_and_clean(synthetic_csv)
    assert stats.dropped_missing == 1
    assert stats.dropped_duplicates == 1
    assert stats.dropped_invalid_label == 1
    assert stats.total_rows == 70
    assert set(df["status"].unique()) <= set(VALID_LABELS)


def test_load_and_clean_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_and_clean("/nonexistent/path/data.csv")


def test_load_and_clean_missing_columns_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_and_clean(str(bad_csv))


def test_load_and_clean_empty_dataset_path_raises():
    with pytest.raises(FileNotFoundError):
        load_and_clean("")


def test_prepare_dataset_splits_and_formats(synthetic_csv, fake_tokenizer):
    train_data, test_data, stats = prepare_dataset(
        synthetic_csv, fake_tokenizer, test_size=0.2, seed=42
    )
    assert len(train_data) + len(test_data) == stats.total_rows
    assert train_data.column_names == ["text"]
    assert set(test_data.column_names) >= {"statement", "status", "text"}


def test_prepare_dataset_train_test_no_overlap(synthetic_csv, fake_tokenizer):
    """No test-set statement should leak into any training example's formatted
    text — this is the actual data-leakage check the pipeline must satisfy."""
    train_data, test_data, _ = prepare_dataset(synthetic_csv, fake_tokenizer, test_size=0.2, seed=42)
    train_texts = train_data["text"]
    for statement in test_data["statement"]:
        assert not any(statement in t for t in train_texts), (
            f"Test statement leaked into a training example: {statement!r}"
        )


def test_prepare_dataset_is_deterministic_given_seed(synthetic_csv, fake_tokenizer):
    _, test_data_a, _ = prepare_dataset(synthetic_csv, fake_tokenizer, test_size=0.2, seed=42)
    _, test_data_b, _ = prepare_dataset(synthetic_csv, fake_tokenizer, test_size=0.2, seed=42)
    assert list(test_data_a["statement"]) == list(test_data_b["statement"])


def test_prepare_dataset_stratifies_by_label(synthetic_csv, fake_tokenizer):
    """With 10 examples per label and test_size=0.2, each label should
    contribute ~2 examples to the test split (stratified), not a skewed mix."""
    _, test_data, _ = prepare_dataset(synthetic_csv, fake_tokenizer, test_size=0.2, seed=42)
    counts = Counter(test_data["status"])
    assert set(counts.keys()) == set(VALID_LABELS)
    for label, count in counts.items():
        assert count == 2, f"Expected ~2 test examples for {label}, got {count}"
