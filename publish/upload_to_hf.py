#!/usr/bin/env python3
"""Upload VIDAR-Grounded-7B (GGUF) + model card to the Hugging Face Hub.
Usage: HF_TOKEN=hf_xxx python upload_to_hf.py <hf_handle>/VIDAR-Grounded-7B [--private]
"""
import sys, os
from huggingface_hub import HfApi, create_repo

repo_id = sys.argv[1] if len(sys.argv) > 1 else None
if not repo_id or "/" not in repo_id:
    sys.exit("give a repo id: <hf_handle>/VIDAR-Grounded-7B")
private = "--private" in sys.argv

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF = os.path.join(HERE, "..", "cloud-artifacts", "shootout-full-2026-06-23",
                    "blobs", "job_d6da0f0ddf234fb4bf13", "qwen25-7b-r16-Q4_K_M.gguf")
README = os.path.join(HERE, "README.md")
assert os.path.exists(GGUF), f"missing GGUF: {GGUF}"

api = HfApi()
create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
print(f"[hf] uploading README -> {repo_id}")
api.upload_file(path_or_fileobj=README, path_in_repo="README.md", repo_id=repo_id)
print(f"[hf] uploading GGUF ({os.path.getsize(GGUF)/1e9:.1f} GB) -> {repo_id}")
api.upload_file(path_or_fileobj=GGUF, path_in_repo="VIDAR-Grounded-7B-Q4_K_M.gguf", repo_id=repo_id)
print(f"[hf] done: https://huggingface.co/{repo_id}")
