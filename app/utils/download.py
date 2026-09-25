import argparse
import logging

from huggingface_hub import login, snapshot_download

from app.configs.settings import get_settings
from app.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def download_model(hf_token: str, hf_repo_id: str, cache_dir: str = "./models") -> str:
    """Authenticates and downloads a base model snapshot from Hugging Face Hub."""
    login(token=hf_token)

    local_dir = snapshot_download(
        repo_id=hf_repo_id,
        revision="main",
        cache_dir=cache_dir,
    )

    logger.info("Model downloaded to: %s", local_dir)
    return local_dir


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Download the configured base model from HF Hub.")
    parser.add_argument("--model-id", default=None, help="Override the model id to download.")
    args = parser.parse_args()

    settings = get_settings()
    model_id = args.model_id or settings.model_id
    download_model(
        hf_token=settings.require_hf_token(),
        hf_repo_id=model_id,
        cache_dir=settings.local_model_cache,
    )
