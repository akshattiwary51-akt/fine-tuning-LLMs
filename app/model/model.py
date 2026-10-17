"""
## **Quantization Process**

Quantization reduces memory usage by dividing the range of possible weights into
discrete bins and storing the bin index instead of the full floating-point
value. The range of weights, often normalized to a fixed range (e.g.,
[-1.0, 1.0]), is divided into bins. For example, with **4-bit quantization**,
the range is split into 2^4 = 16 bins, each with a center value. During
quantization, each weight is assigned to its closest bin center, replacing
the original value. Instead of storing the weight itself, only the bin index
(e.g., an integer between 0 and 15) is saved. This dramatically reduces the
memory footprint while maintaining approximate accuracy.

**Note on Training (QLoRA):** In this project, the quantized base weights are **frozen**
(not updated). During backpropagation, gradients flow through these frozen weights but
only accumulate and update the **LoRA adapter parameters**—the small trainable matrices
injected into the model layers.

---

## Backends

Two code paths are supported, selected by `Settings.use_unsloth`:

- **Unsloth**: `FastLanguageModel` handles 4-bit loading internally and provides a
  fused, memory-optimized training path.
- **Standard Hugging Face**: `AutoModelForCausalLM` + `BitsAndBytesConfig` + `peft`.

Both paths now read their LoRA hyperparameters (rank, alpha, dropout, target
modules) from the single `app.configs.lora_config.get_lora_spec()` function, so
switching `use_unsloth` on/off no longer silently changes what LoRA/QLoRA mean
(previously, the Unsloth branch hardcoded `lora_dropout=0` and never passed
`target_modules`, so it did not train the same set of parameters as the
standard path).
"""

from __future__ import annotations

import logging

import torch
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from app.configs.lora_config import get_lora_spec, to_peft_config
from app.configs.settings import Settings
from app.utils.environment import supports_bf16

logger = logging.getLogger(__name__)


def count_trainable_parameters(model) -> dict:
    """Returns trainable/total parameter counts and the trainable percentage.

    Phase 11 (resource-efficient fine-tuning) asks this project to expose real
    numbers rather than just claim efficiency; this is what `run_metadata.json`
    reports it from (see app/train.py). Uses peft's own accounting
    (`get_nb_trainable_parameters`) when available, since it correctly handles
    quantized/frozen base-model parameters; falls back to a manual count over
    `model.parameters()` for any other model type.
    """
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable_params, total_params = model.get_nb_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

    trainable_pct = 100 * trainable_params / total_params if total_params else 0.0
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_pct, 4),
    }


def load_model(settings: Settings, inference: bool = False):
    """Loads the base model and either attaches existing adapters (inference)
    or initializes new trainable adapters (training).

    Args:
        settings: resolved pipeline configuration (model id, LoRA mode, backend,
            sequence length, adapter directory all come from here).
        inference: if True, loads `settings.adapter_dir` onto the base model and
            switches to eval mode; if False, initializes fresh adapters for training.
    """
    model_id = settings.model_id
    lora_spec = get_lora_spec(settings.lora_mode)

    if settings.use_unsloth:
        return _load_unsloth(settings, model_id, lora_spec, inference)
    return _load_standard(settings, model_id, lora_spec, inference)


def _load_unsloth(settings: Settings, model_id: str, lora_spec, inference: bool):
    # By this point `unsloth` has already been imported at the top of the
    # calling entrypoint (app/train.py / app/inference.py), before
    # transformers/peft/trl — this repeated import is just for local
    # readability in this function and is a no-op (Python caches modules).
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    logger.info("[Unsloth] Loading base model '%s' (inference=%s)...", model_id, inference)

    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=settings.max_seq_length,  # was hardcoded to 2048, now a single config value
        dtype=None,
        load_in_4bit=(settings.lora_mode == "qlora"),
    )

    if inference:
        logger.info("[Unsloth] Loading existing adapters from %s...", settings.adapter_dir)
        model = PeftModel.from_pretrained(model, settings.adapter_dir)
        FastLanguageModel.for_inference(model)
        return model

    logger.info("[Unsloth] Initializing new adapters for training (%s)...", settings.lora_mode)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_spec.r,
        lora_alpha=lora_spec.lora_alpha,
        lora_dropout=lora_spec.lora_dropout,  # bug fix: was hardcoded to 0
        target_modules=lora_spec.target_modules,  # bug fix: was never passed at all
        finetune_vision_layers=False,  # Keep vision part frozen
        finetune_language_layers=True,  # Train the language part
        finetune_attention_modules=True,  # Train attention
        finetune_mlp_modules=True,  # Train MLPs
        bias=lora_spec.bias,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    model.print_trainable_parameters()
    logger.info("[Unsloth] Model prepared successfully (training mode).")
    return model


def _load_standard(settings: Settings, model_id: str, lora_spec, inference: bool):
    if settings.lora_mode == "qlora":
        # Compute dtype matches the actual training precision (see app/train.py,
        # which picks bf16 vs fp16 the same way via supports_bf16()) rather than
        # being hardcoded — bf16 needs Ampere+ GPUs; older GPUs (e.g. Colab's
        # free-tier T4) only support fp16, and a mismatch here would be silently
        # inconsistent even where it doesn't outright fail.
        compute_dtype = torch.bfloat16 if supports_bf16() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    logger.info("[Standard] Loading base model '%s' (inference=%s)...", model_id, inference)
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        quantization_config=bnb_config,
        attn_implementation="eager",
        dtype="auto",
        device_map="auto",
    )

    if inference:
        logger.info("[Standard] Loading existing adapters from %s...", settings.adapter_dir)
        model = PeftModel.from_pretrained(base_model, settings.adapter_dir)
        model.eval()
        logger.info("[Standard] Model loaded successfully (inference mode).")
        return model

    logger.info("[Standard] Initializing new adapters for training (%s)...", settings.lora_mode)
    if settings.lora_mode == "qlora":
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model.enable_input_require_grads()

    peft_config = to_peft_config(lora_spec)
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    logger.info("[Standard] Model prepared successfully (training mode).")
    return model
