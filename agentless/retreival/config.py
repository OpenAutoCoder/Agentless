import os
from pathlib import Path

GITHUB_PERSONAL_TOKEN = os.getenv("GITHUB_PERSONAL_TOKEN")

WORKING_DIR = Path(os.getenv("WORKING_DIR", "/tmp/swe-bench-verified"))
if not WORKING_DIR.exists():
    WORKING_DIR.mkdir(parents=True)

EMBEDDINGS_DIR = WORKING_DIR / "embeddings"
if not EMBEDDINGS_DIR.exists():
    EMBEDDINGS_DIR.mkdir(parents=True)


GITHUB_REPOS_DIR = WORKING_DIR / "github_repos"
if not GITHUB_REPOS_DIR.exists():
    GITHUB_REPOS_DIR.mkdir(parents=True)
