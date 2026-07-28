from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
)

REPO_ID = "Cthuunghiacity/nc-ai-money-worker"

files = {
    "app.py": Path("autocap_worker_patch/app.py"),
    "autocap_routes.py": Path(
        "autocap_worker_patch/autocap_routes.py"
    ),
    "requirements.txt": Path(
        "autocap_worker_patch/requirements.txt"
    ),
    "Dockerfile": Path(
        "autocap_worker_patch/Dockerfile"
    ),
}

for destination, source in files.items():
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(
            f"MISSING_DEPLOY_FILE: {source}"
        )

token = os.environ.get("HF_TOKEN", "").strip()
owner_key = os.environ.get(
    "AUTOCAP_KEY",
    "",
).strip()

if not token:
    raise RuntimeError("HF_TOKEN_EMPTY")

if not owner_key:
    raise RuntimeError("AUTOCAP_KEY_EMPTY")

api = HfApi(token=token)

operations = [
    CommitOperationAdd(
        path_in_repo=destination,
        path_or_fileobj=str(source),
    )
    for destination, source in files.items()
]

result = api.create_commit(
    repo_id=REPO_ID,
    repo_type="space",
    operations=operations,
    commit_message=(
        "deploy NAMI AutoCap CTranslate2 worker"
    ),
)

api.add_space_secret(
    repo_id=REPO_ID,
    key="NAMI_AUTOCAP_KEY",
    value=owner_key,
)

print(f"HF_COMMIT_URL={result.commit_url}")
print("AUTOCAP_MODULE_UPLOADED=PASS")
print("AUTOCAP_SECRET_CONFIGURED=PASS")
