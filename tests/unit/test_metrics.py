import pytest

from app.data.prompts import VALID_LABELS
from app.evaluation.metrics import compute_metrics


def test_perfect_predictions_score_one():
    y_true = ["Normal", "Suicidal", "Anxiety"]
    y_pred = ["Normal", "Suicidal", "Anxiety"]
    result = compute_metrics(y_true, y_pred)
    assert result.accuracy == 1.0
    assert result.macro_f1 == pytest.approx(1.0 / len(VALID_LABELS) * 3)  # only 3 classes have support
    for label in ("Normal", "Suicidal", "Anxiety"):
        assert result.per_class[label]["f1"] == 1.0


def test_all_wrong_predictions_score_zero_accuracy():
    y_true = ["Normal", "Normal", "Normal"]
    y_pred = ["Suicidal", "Suicidal", "Suicidal"]
    result = compute_metrics(y_true, y_pred)
    assert result.accuracy == 0.0
    assert result.per_class["Normal"]["recall"] == 0.0
    assert result.per_class["Suicidal"]["precision"] == 0.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_metrics(["Normal"], ["Normal", "Anxiety"])


def test_unparseable_predictions_counted_separately():
    y_true = ["Normal", "Normal"]
    y_pred = ["Normal", "not a real label"]
    result = compute_metrics(y_true, y_pred)
    assert result.unparseable_predictions == 1
    # The unparseable prediction should count against recall for its true class,
    # not silently disappear or count as a false positive for some other class.
    assert result.per_class["Normal"]["recall"] == 0.5


def test_accuracy_alone_can_hide_imbalance():
    """Demonstrates why this project doesn't evaluate on accuracy alone: a
    majority-class-only predictor scores well on accuracy but terribly on
    macro F1 for minority classes."""
    y_true = ["Normal"] * 90 + ["Suicidal"] * 10
    y_pred = ["Normal"] * 100  # always predicts the majority class
    result = compute_metrics(y_true, y_pred)
    assert result.accuracy == pytest.approx(0.9)
    assert result.per_class["Suicidal"]["recall"] == 0.0
    assert result.per_class["Suicidal"]["f1"] == 0.0
    assert result.macro_f1 < result.accuracy


def test_confusion_matrix_diagonal_matches_true_positives():
    y_true = ["Normal", "Normal", "Suicidal"]
    y_pred = ["Normal", "Suicidal", "Suicidal"]
    result = compute_metrics(y_true, y_pred)
    assert result.confusion_matrix["Normal"]["Normal"] == 1
    assert result.confusion_matrix["Normal"]["Suicidal"] == 1
    assert result.confusion_matrix["Suicidal"]["Suicidal"] == 1


def test_human_summary_and_json_do_not_error():
    result = compute_metrics(["Normal"], ["Normal"])
    assert "Accuracy" in result.human_summary()
    assert '"accuracy"' in result.to_json()


def test_save_writes_json_file(tmp_path):
    result = compute_metrics(["Normal"], ["Normal"])
    out_path = tmp_path / "nested" / "metrics.json"
    result.save(str(out_path))
    assert out_path.exists()
    assert "accuracy" in out_path.read_text()
