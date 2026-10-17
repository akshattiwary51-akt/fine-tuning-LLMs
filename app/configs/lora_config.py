"""
LoRA (Low-Rank Adaptation) keeps the vast majority of the original model frozen
and introduces a tiny subset of new, trainable parameters (the LoRA adapters).
Only this small subset is updated during fine-tuning, saving computation and memory.

Both `lora_cfg` and `qlora_cfg` are exposed as `peft.LoraConfig` objects (used by
the standard Hugging Face path) AND as plain dataclasses via `get_lora_spec()`
(used by the Unsloth path, which takes its own kwargs rather than a LoraConfig
object).

Previously, the Unsloth branch in app/model/model.py hardcoded lora_dropout=0 and
never passed target_modules at all, meaning LoRA/QLoRA behaved differently
depending on which backend was active. `get_lora_spec()` is the single source of
truth both branches now read from, so that no longer happens.
"""

from dataclasses import dataclass

from peft import LoraConfig

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",  # Attention projections: query, key, value
    "o_proj",  # Attention output projection
    "gate_proj",
    "up_proj",
    "down_proj",  # MLP gating + up / down projections
]


@dataclass(frozen=True)
class LoraSpec:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    bias: str = "none"
    use_rslora: bool = False


LORA_SPEC = LoraSpec(
    r=8,  # Rank: size of the low-rank matrices (↑ r ⇒ ↑ capacity & VRAM usage)
    lora_alpha=16,  # Scaling factor for LoRA updates (often >= r; controls update magnitude)
    lora_dropout=0.05,  # Dropout on LoRA layers
    target_modules=TARGET_MODULES,
)

QLORA_SPEC = LoraSpec(
    r=64,  # LoRA rank — higher is normal in QLoRA (32-128 typical)
    lora_alpha=128,  # Scaling factor (2x r)
    lora_dropout=0.1,  # Dropout improves generalization for QLoRA
    target_modules=TARGET_MODULES,
)


def get_lora_spec(lora_mode: str) -> LoraSpec:
    if lora_mode == "lora":
        return LORA_SPEC
    if lora_mode == "qlora":
        return QLORA_SPEC
    raise ValueError(f"Invalid lora_mode: '{lora_mode}'. Must be 'lora' or 'qlora'.")


def to_peft_config(spec: LoraSpec) -> LoraConfig:
    """Builds a peft.LoraConfig for the standard (non-Unsloth) Hugging Face path."""
    return LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=spec.target_modules,
        r=spec.r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        bias=spec.bias,
        use_rslora=spec.use_rslora,
    )


# Backwards-compatible peft.LoraConfig objects (still usable directly if needed).
lora_cfg = to_peft_config(LORA_SPEC)
qlora_cfg = to_peft_config(QLORA_SPEC)
