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


def test_hf_token_env_var_takes_priority_over_colab(monkeypatch):
    """If HF_TOKEN is set as a normal env var, the Colab secrets lookup should
    never even be attempted."""
    from app.configs.settings import _get_hf_token_from_env_or_colab

    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert _get_hf_token_from_env_or_colab() == "env-token"


def test_hf_token_falls_back_to_colab_secrets_when_env_unset(monkeypatch):
    """When HF_TOKEN isn't set as an env var but a `google.colab` module is
    importable (i.e. we're running in Colab), fall back to its secret manager."""
    import sys
    import types

    from app.configs.settings import _get_hf_token_from_env_or_colab

    monkeypatch.delenv("HF_TOKEN", raising=False)

    fake_google_colab = types.ModuleType("google.colab")
    fake_userdata = types.ModuleType("google.colab.userdata")
    fake_userdata.get = lambda key: "colab-secret-token" if key == "HF_TOKEN" else None
    fake_google_colab.userdata = fake_userdata

    monkeypatch.setitem(sys.modules, "google.colab", fake_google_colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", fake_userdata)

    assert _get_hf_token_from_env_or_colab() == "colab-secret-token"


def test_hf_token_returns_none_outside_colab_without_env_var(monkeypatch):
    from app.configs.settings import _get_hf_token_from_env_or_colab

    monkeypatch.delenv("HF_TOKEN", raising=False)
    # No google.colab module available (the normal case outside a notebook) —
    # the import inside the function will raise ImportError/ModuleNotFoundError
    # and should be swallowed, returning None rather than propagating.
    assert _get_hf_token_from_env_or_colab() is None
