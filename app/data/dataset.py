"""
Dataset loading, cleaning, validation and formatting for the SFT pipeline.

Replaces app/utils/data_prep.py with: schema validation, duplicate detection,
label validation, per-class distribution stats (important — this dataset is
imbalanced), and a *stratified* train/test split so imbalance is preserved
proportionally rather than by chance. Splitting happens strictly before any
prompt formatting or tokenization, so there is no leakage of train text into
test formatting logic or vice versa.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from app.data.prompts import VALID_LABELS, format_training_example

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("statement", "status")


@dataclass
class DatasetStats:
    """Summary statistics captured while cleaning the raw CSV.

    Kept lightweight and JSON/print-friendly so it can be logged, saved
    alongside a training run, or asserted on in tests.
    """

    total_rows: int
    dropped_missing: int
    dropped_duplicates: int
    dropped_invalid_label: int
    class_distribution: dict[str, int]

    def summary(self) -> str:
        lines = [
            f"Rows kept: {self.total_rows}",
            f"Dropped (missing statement/status): {self.dropped_missing}",
            f"Dropped (exact duplicates): {self.dropped_duplicates}",
            f"Dropped (label outside {VALID_LABELS}): {self.dropped_invalid_label}",
            "Class distribution:",
        ]
        for label, count in sorted(self.class_distribution.items(), key=lambda kv: -kv[1]):
            pct = 100 * count / self.total_rows if self.total_rows else 0.0
            lines.append(f"  {label:<22} {count:>6}  ({pct:5.1f}%)")
        return "\n".join(lines)


def normalize_text(text) -> str:
    """Sanitizes input text by removing newlines/tabs and surrounding whitespace."""
    if not isinstance(text, str):
        return ""
    return text.replace("\n", " ").replace("\t", " ").strip()


def load_and_clean(dataset_path: str) -> tuple[pd.DataFrame, DatasetStats]:
    """Loads the raw CSV and applies validation/cleaning.

    Steps (in order, each counted so nothing is silently dropped):
      1. Required-column check (fails loudly if the CSV doesn't look right).
      2. Drop rows with a missing statement or status.
      3. Normalize whitespace in text fields.
      4. Drop exact duplicate (statement, status) rows.
      5. Drop rows whose label isn't one of the seven known classes — this
         catches typos/garbage labels rather than silently training on them.
    """
    path = Path(dataset_path)
    if not dataset_path or not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set DATASET_PATH in your .env "
            "file (see .env.example) or pass --dataset-path on the command line."
        )

    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    total_before = len(df)

    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    dropped_missing = total_before - len(df)

    df["statement"] = df["statement"].apply(normalize_text)
    df["status"] = df["status"].apply(normalize_text)
    df = df[df["statement"].str.len() > 0]

    before_dedup = len(df)
    df = df.drop_duplicates(subset=list(REQUIRED_COLUMNS))
    dropped_duplicates = before_dedup - len(df)

    before_label_filter = len(df)
    df = df[df["status"].isin(VALID_LABELS)]
    dropped_invalid_label = before_label_filter - len(df)

    class_distribution = df["status"].value_counts().to_dict()

    stats = DatasetStats(
        total_rows=len(df),
        dropped_missing=dropped_missing,
        dropped_duplicates=dropped_duplicates,
        dropped_invalid_label=dropped_invalid_label,
        class_distribution=class_distribution,
    )
    return df.reset_index(drop=True), stats


def prepare_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset, DatasetStats]:
    """Loads, cleans, validates, stratified-splits and formats the dataset.

    Returns (train_data, test_data, stats). `train_data` has only the
    tokenizer-formatted "text" column (what SFTTrainer needs); `test_data`
    keeps the raw "statement"/"status" columns too, since inference/evaluation
    needs the original statement and ground-truth label, not the pre-baked
    training string.

    The split is stratified by label so the (imbalanced) class proportions in
    train and test match the full dataset's proportions rather than drifting
    by chance — important with several minority classes.
    """
    df, stats = load_and_clean(dataset_path)
    logger.info("Dataset loaded and cleaned:\n%s", stats.summary())

    # Stratified split done directly in pandas (per-class groupby + sample)
    # rather than via `datasets.class_encode_column`/`stratify_by_column`:
    # that route round-trips "status" through an integer ClassLabel and back,
    # which is fragile (dtype surprises after `.map`) for no real benefit
    # here since we already have the labels as plain strings. This keeps
    # "status" a string throughout and is deterministic given `seed`.
    test_frames = []
    train_frames = []
    for _, group in df.groupby("status", group_keys=False):
        test_group = group.sample(frac=test_size, random_state=seed)
        train_group = group.drop(test_group.index)
        test_frames.append(test_group)
        train_frames.append(train_group)

    train_df = pd.concat(train_frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    train_data = train_dataset.map(
        format_training_example,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Applying chat template (train)",
    ).remove_columns(["statement", "status"])

    test_data = test_dataset.map(
        format_training_example,
        fn_kwargs={"tokenizer": tokenizer},
        desc="Applying chat template (test)",
    )

    logger.info(
        "Split into %d train / %d test examples (test_size=%.2f, seed=%d, stratified by label).",
        len(train_data), len(test_data), test_size, seed,
    )

    return train_data, test_data, stats
