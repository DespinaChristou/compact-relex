import os
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import HfApi


@dataclass(frozen=True)
class HFConfig:
    token_env: str
    org_or_user: str
    private: bool


def get_hf_token(token_env: str) -> str:
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Missing Hugging Face token. Please set env var {token_env} (e.g., export {token_env}=...)."
        )
    return token


def ensure_private_model_repo(repo_id: str, token: str) -> None:
    """
    Create the repo if needed, ensure it's private.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)


def push_model_dir(local_dir: str, repo_id: str, token: str, commit_message: Optional[str] = None) -> None:
    """
    Push a local folder containing a HF Transformers model (and tokenizer) to a private repo.
    """
    api = HfApi(token=token)
    ensure_private_model_repo(repo_id=repo_id, token=token)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=local_dir,
        commit_message=commit_message or "Upload model artifacts",
    )
