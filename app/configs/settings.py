"""
Centralized, validated configuration for the fine-tuning pipeline.

Values are resolved in this order (highest priority first):
    1. Explicit keyword arguments passed to `Settings(...)` / `get_settings(...)`
    2. Environment variables (see field docs below, and `.env.example`)
    3. Built-in defaults

Design notes:
- Secrets (HF_TOKEN) are read from the environment but are NOT required just to
  construct a `Settings` object. They're only enforced at the point of use
  (upload/download), with an actionable error message. This keeps the config
  object usable in unit tests that never touch Hugging Face Hub.
- `run_name` / `adapter_dir` encode both the LoRA mode AND whether Unsloth was
  used, so switching backends never silently overwrites a previous run's
  output (this was a real bug in the original single hardcoded ADAPTER_DIR).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VALID_LORA_MODES = ("lora", "qlora")


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_hf_token_from_env_or_colab() -> str | None:
    """Resolves HF_TOKEN from, in order: the environment (.env / shell), then
    Google Colab's secret manager if running in Colab and the env var isn't set.

    Colab's `userdata` secrets (set via the key icon in the left sidebar) are
    the recommended way to store tokens in a notebook — safer than writing a
    plaintext .env into ephemeral notebook storage. This is a no-op outside
    Colab (the import simply fails and is caught).
    """
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get("HF_TOKEN")
    except Exception:
        return None


@dataclass
class Settings:
    # --- Secrets (never defaulted, only validated on use) -------------
    hf_token: str | None = field(default_factory=_get_hf_token_from_env_or_colab)

    # --- Model selection ------------------------------------------------
    use_unsloth: bool = field(default_factory=lambda: _env_bool("USE_UNSLOTH", True))
    lora_mode: str = field(default_factory=lambda: os.getenv("LORA_MODE", "qlora"))
    base_model_hf: str = field(default_factory=lambda: os.getenv("BASE_MODEL_HF", "google/gemma-3-1b-it"))
    base_model_unsloth: str = field(
        default_factory=lambda: os.getenv("BASE_MODEL_UNSLOTH", "unsloth/gemma-3-1b-it")
    )

    # --- Data -----------------------------------------------------------
    dataset_path: str = field(default_factory=lambda: os.getenv("DATASET_PATH", ""))
    test_size: float = field(default_factory=lambda: float(os.getenv("TEST_SIZE", "0.2")))
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))

    # --- Sequence length (single source of truth — used by model loading
    # AND by SFTConfig, so they can never drift apart like they did before) --
    max_seq_length: int = field(default_factory=lambda: int(os.getenv("MAX_SEQ_LENGTH", "1024")))

    # --- Paths ------------------------------------------------------------
    output_root: str = field(default_factory=lambda: os.getenv("OUTPUT_ROOT", "./sft_output"))
    local_model_cache: str = field(default_factory=lambda: os.getenv("LOCAL_MODEL_CACHE", "./models"))

    # --- Hugging Face Hub upload target -----------------------------------
    hf_username: str | None = field(default_factory=lambda: os.getenv("HF_USERNAME"))
    hf_personal_repo_id_override: str | None = field(default_factory=lambda: os.getenv("HF_PERSONAL_REPO_ID"))

    def __post_init__(self) -> None:
        if self.lora_mode not in VALID_LORA_MODES:
            raise ValueError(f"Invalid LORA_MODE '{self.lora_mode}'. Must be one of {VALID_LORA_MODES}.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"TEST_SIZE must be strictly between 0 and 1, got {self.test_size}.")
        if self.max_seq_length <= 0:
            raise ValueError(f"MAX_SEQ_LENGTH must be a positive integer, got {self.max_seq_length}.")
        if self.seed < 0:
            raise ValueError(f"SEED must be non-negative, got {self.seed}.")

    # --- Derived / computed properties -------------------------------------
    @property
    def model_id(self) -> str:
        """HF repo id of the base model to load, depending on the active backend."""
        return self.base_model_unsloth if self.use_unsloth else self.base_model_hf

    @property
    def backend_name(self) -> str:
        return "unsloth" if self.use_unsloth else "hf"

    @property
    def run_name(self) -> str:
        """Unique identifier for this experiment configuration.

        Encodes model + LoRA mode + backend so different configurations never
        overwrite each other's checkpoints/adapters.
        """
        safe_model = self.model_id.replace("/", "__")
        return f"{safe_model}-{self.lora_mode}-{self.backend_name}"

    @property
    def run_output_dir(self) -> str:
        return str(Path(self.output_root) / self.run_name)

    @property
    def adapter_dir(self) -> str:
        return str(Path(self.run_output_dir) / "adapters")

    def require_hf_token(self) -> str:
        """Fetches the HF token, raising an actionable error if it's missing.

        Only called at the point HF Hub auth is actually needed (download/upload),
        not at Settings construction time.
        """
        if not self.hf_token:
            raise ValueError(
                "Missing HF_TOKEN. Set it in your .env file (see .env.example) or as an "
                "environment variable. Get a token at https://huggingface.co/settings/tokens"
            )
        return self.hf_token

    def require_dataset_path(self) -> str:
        if not self.dataset_path:
            raise ValueError(
                "DATASET_PATH is not set. Set it in your .env file or pass "
                "--dataset-path on the command line."
            )
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.dataset_path}'. Check DATASET_PATH / --dataset-path."
            )
        return self.dataset_path

    @property
    def hf_personal_repo_id(self) -> str:
        """Destination repo for uploading trained adapters.

        Resolution order: explicit HF_PERSONAL_REPO_ID env var > HF_USERNAME-based
        default > actionable error. Unlike the original code, this never silently
        defaults to the original author's personal namespace.
        """
        if self.hf_personal_repo_id_override:
            return self.hf_personal_repo_id_override
        if self.hf_username:
            return f"{self.hf_username}/gemma-3-1b-it-{self.lora_mode}-{self.backend_name}"
        raise ValueError(
            "Cannot determine the upload destination repo. Set HF_PERSONAL_REPO_ID "
            "(full 'namespace/repo-name') or HF_USERNAME in your .env file."
        )


def get_settings(**overrides) -> Settings:
    """Factory for building a Settings object, optionally overriding fields.

    Useful for CLI argument overrides and for unit tests, e.g.:
        get_settings(lora_mode="lora", dataset_path="tests/fixtures/sample.csv")
    """
    return Settings(**overrides)
