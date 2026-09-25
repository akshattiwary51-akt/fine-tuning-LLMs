import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(model_id: str, inference: bool = False) -> AutoTokenizer:
    """Loads and configures the tokenizer for the given model id."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if inference:
        # Inference (generation): pad on the LEFT so the prompt ends right
        # before generation starts (no gap between prompt and new tokens).
        tokenizer.padding_side = "left"
    else:
        # Training: pad on the RIGHT so the model learns to predict real next
        # tokens, not padding.
        tokenizer.padding_side = "right"

    # Gemma/Llama tokenizers have no pad token by default; fall back to EOS to
    # avoid crashes during batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def inspect_tokenizer(tokenizer: AutoTokenizer, sample_text: str = "What is AI?") -> None:
    """Logs tokenizer details to verify it behaves as expected. Useful for
    interactive debugging; not called during normal training/inference runs."""
    tokens = tokenizer.tokenize(sample_text)
    input_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded_text = tokenizer.decode(input_ids)

    logger.info("Input Text: '%s'", sample_text)
    logger.info("Tokens: %s", tokens)
    logger.info("Token IDs: %s", input_ids)
    logger.info("Decoded: '%s'", decoded_text)
    logger.info("BOS Token ID: %s (%s)", tokenizer.bos_token_id, tokenizer.bos_token)
    logger.info("EOS Token ID: %s (%s)", tokenizer.eos_token_id, tokenizer.eos_token)
    logger.info("PAD Token ID: %s (%s)", tokenizer.pad_token_id, tokenizer.pad_token)
    logger.info("Padding Side: %s", tokenizer.padding_side)
    logger.info("Model Max Length: %s", tokenizer.model_max_length)
