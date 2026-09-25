import argparse
import logging

from huggingface_hub import HfApi, login

from app.configs.settings import get_settings
from app.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def upload_model(hf_token: str, hf_personal_repo_id: str, local_model_dir: str) -> str:
    """Authenticates and uploads a local adapter directory to a HF Hub repository."""
    login(token=hf_token)

    api = HfApi()
    api.create_repo(repo_id=hf_personal_repo_id, exist_ok=True)
    logger.info("Starting upload from '%s' to '%s'...", local_model_dir, hf_personal_repo_id)

    upload_info = api.upload_folder(
        folder_path=local_model_dir,
        repo_id=hf_personal_repo_id,
        repo_type="model",
        commit_message="Upload model via script",
    )

    logger.info("Upload complete! View at: %s", upload_info)
    return upload_info


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Upload trained adapters to HF Hub.")
    parser.add_argument("--repo-id", default=None, help="Override the destination repo id.")
    parser.add_argument("--adapter-dir", default=None, help="Override the local adapter directory.")
    args = parser.parse_args()

    settings = get_settings()
    upload_model(
        hf_token=settings.require_hf_token(),
        hf_personal_repo_id=args.repo_id or settings.hf_personal_repo_id,
        local_model_dir=args.adapter_dir or settings.adapter_dir,
    )
