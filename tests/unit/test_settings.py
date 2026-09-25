import pytest

from app.configs.settings import get_settings


def test_defaults_do_not_require_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    settings = get_settings()
    assert settings.hf_token is None


def test_require_hf_token_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    settings = get_settings()
    with pytest.raises(ValueError, match="HF_TOKEN"):
        settings.require_hf_token()


def test_require_hf_token_succeeds_when_set():
    settings = get_settings(hf_token="hf_fake_token_for_testing")
    assert settings.require_hf_token() == "hf_fake_token_for_testing"


@pytest.mark.parametrize("bad_mode", ["LORA", "q-lora", "", "banana"])
def test_invalid_lora_mode_raises(bad_mode):
    with pytest.raises(ValueError, match="LORA_MODE"):
        get_settings(lora_mode=bad_mode)


@pytest.mark.parametrize("bad_test_size", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_size_raises(bad_test_size):
    with pytest.raises(ValueError, match="TEST_SIZE"):
        get_settings(test_size=bad_test_size)


def test_invalid_max_seq_length_raises():
    with pytest.raises(ValueError, match="MAX_SEQ_LENGTH"):
        get_settings(max_seq_length=0)


def test_model_id_switches_with_unsloth_flag():
    with_unsloth = get_settings(use_unsloth=True)
    without_unsloth = get_settings(use_unsloth=False)
    assert with_unsloth.model_id == with_unsloth.base_model_unsloth
    assert without_unsloth.model_id == without_unsloth.base_model_hf


def test_run_name_differs_across_lora_mode_and_backend():
    """Regression test for the original bug where ADAPTER_DIR was a single
    hardcoded path shared by every configuration, silently overwriting
    previous runs when the LoRA mode or backend changed."""
    names = {
        get_settings(lora_mode="lora", use_unsloth=True).run_name,
        get_settings(lora_mode="qlora", use_unsloth=True).run_name,
        get_settings(lora_mode="lora", use_unsloth=False).run_name,
        get_settings(lora_mode="qlora", use_unsloth=False).run_name,
    }
    assert len(names) == 4, "Each (lora_mode, backend) combination must get a distinct run_name"


def test_adapter_dir_derives_from_run_name():
    from pathlib import Path

    settings = get_settings()
    assert settings.run_name in settings.adapter_dir
    # Compare via Path (not raw string prefix) since Path() normalizes "./x" to "x".
    assert Path(settings.adapter_dir).is_relative_to(Path(settings.output_root))


def test_require_dataset_path_missing_file(tmp_path):
    settings = get_settings(dataset_path=str(tmp_path / "does_not_exist.csv"))
    with pytest.raises(FileNotFoundError):
        settings.require_dataset_path()


def test_require_dataset_path_empty_string():
    settings = get_settings(dataset_path="")
    with pytest.raises(ValueError, match="DATASET_PATH"):
        settings.require_dataset_path()


def test_hf_personal_repo_id_requires_explicit_config():
    settings = get_settings(hf_username=None, hf_personal_repo_id_override=None)
    with pytest.raises(ValueError, match="HF_PERSONAL_REPO_ID|HF_USERNAME"):
        _ = settings.hf_personal_repo_id


def test_hf_personal_repo_id_from_username():
    settings = get_settings(hf_username="someuser", hf_personal_repo_id_override=None)
    assert settings.hf_personal_repo_id.startswith("someuser/")


def test_hf_personal_repo_id_explicit_override_wins():
    settings = get_settings(
        hf_username="someuser",
        hf_personal_repo_id_override="explicit-namespace/explicit-repo",
    )
    assert settings.hf_personal_repo_id == "explicit-namespace/explicit-repo"
