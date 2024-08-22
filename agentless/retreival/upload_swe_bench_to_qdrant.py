from datasets import load_dataset
import os
import json
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os
import subprocess
import time
from functools import wraps
import logging
import sys
from typing import Optional
from pathlib import Path
import git
from agentless.retreival import config
from agentless.retreival import gitrepo
from agentless.retreival.VectorDB import QdrantDB
from typing import List

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result, end_time - start_time
    return wrapper

def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]

def clone_repo(owner: str, repo_name: str, base_branch_or_commit: Optional[str] = None, target_dir: Optional[Path] = None) -> Path:
    repo_dir = target_dir or (config.GITHUB_REPOS_DIR / owner / repo_name)
    if not repo_dir.exists():
        repo_url = f"https://{config.GITHUB_PERSONAL_TOKEN}@github.com/{owner}/{repo_name}.git"
        repo = git.Repo.clone_from(repo_url, repo_dir)
    if base_branch_or_commit:
        repo = git.Repo(repo_dir)
        repo.git.reset('--hard')
        repo.git.checkout(base_branch_or_commit)
    return repo_dir

def ensure_tmp_permissions():
    tmp_dir = "/tmp/cintra_wd"
    os.makedirs(tmp_dir, exist_ok=True)
    subprocess.run(["chmod", "-R", "777", tmp_dir], check=True)

@timing_decorator
def load_swe_bench_dataset():
    """
    Load the SWE-bench Lite dataset from HuggingFace or local file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_file = os.path.join(current_dir, 'swe_bench_lite.jsonl')
    if os.path.exists(local_file):
        print("Loading local file")
        return load_jsonl(local_file)
    else:
        print("Loading from HuggingFace")
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        with open(local_file, 'w') as f:
            for item in dataset:
                json.dump(item, f)
                f.write('\n')
        return dataset

@timing_decorator
def init_repo(owner: str, repo_name: str, base_branch_or_commit: str):
    """
    Initialize the repository and generate embeddings for all files.
    """
    # Call this function before using the directory
    ensure_tmp_permissions()
    
    # Construct the repo path in the cache
    repo_path = config.GITHUB_REPOS_DIR / owner / repo_name
    
    if repo_path.exists():
        print(f"Repository {owner}/{repo_name} found in cache")
        repo = gitrepo.GitRepo(repo_path)
        
        # Fetch the latest changes
        repo.repo.git.fetch('--all')
        
        # Checkout the specified branch or commit
        repo.repo.git.checkout(base_branch_or_commit)
    else:
        print(f"Cloning repository {owner}/{repo_name} at {base_branch_or_commit}")
        # Clone directly into the cache directory
        clone_repo(owner, repo_name, base_branch_or_commit, target_dir=repo_path)
        
        # Create a GitRepo object from the newly cloned repository
        repo = gitrepo.GitRepo(repo_path)
    
    return repo

def main(instance_ids: List[str], force_regenerate_embeddings=False, include_chunk_content=True):
    """
    Main function to process the SWE-bench dataset and upload code to Qdrant.

    This function performs the following steps:
    1. Loads the SWE-bench dataset.
    2. Iterates through each row in the dataset.
    3. Initializes the repository for each instance.
    4. Retrieves the code dictionary for the repository.
    5. Creates a QdrantDB instance for vector operations.
    6. Creates a collection in Qdrant if it doesn't exist.
    7. Optionally regenerates embeddings and deletes existing points.
    8. Uploads the code data to the Qdrant database.

    Args:
        force_regenerate_embeddings (bool): If True, force regeneration of embeddings.
        include_chunk_content (bool): If True, include chunk content in the database.

    Returns:
        None
    """
    data, _ = load_swe_bench_dataset()

    # Filter the data to include only the specified instance_ids from the jsonl file
    data_subset = [item for item in data if item['instance_id'] in instance_ids]

    for item in tqdm(data_subset, desc="Evaluating instances"):
        # Extract owner and repo name from the 'repo' field
        owner, repo_name = item['repo'].split('/')
        
        # Use custom_init_repo to initialize the repository
        fs, init_times = init_repo(owner, repo_name, item['base_commit'])

        code_dict = fs.get_code_dict()

        # Filter the code dictionary to only include Python files
        python_code_dict = {file_path: content for file_path, content in code_dict.items() if file_path.endswith('.py')}

        vector_ops = QdrantDB(repo_name, python_code_dict, include_chunk_content=include_chunk_content, regenerate_embeddings=force_regenerate_embeddings)

        # Create the collection if it doesn't exist
        vector_ops.create_collection_if_not_exists()
        if vector_ops.regenerate_embeddings:
            vector_ops.delete_all_points(force=True)
        vector_ops.upload_data_to_db(vector_ops.codebase_dict)

if __name__ == "__main__":
    import argparse
    import sys
    import logging
    
    parser = argparse.ArgumentParser(description="Evaluate RelevantCodeFinder")
    parser.add_argument("--force-regenerate-embeddings", action="store_true", help="Force regeneration of embeddings")
    parser.add_argument("--include-chunk-content", action="store_true", help="Include chunk content in the database")
    args = parser.parse_args()

    # Filter the dataset to include only the subset of cases from generator.py
    instance_ids_to_upload = [
        'astropy__astropy-12907',
        'sympy__sympy-20590',
        'django__django-10924',
        'matplotlib__matplotlib-23476',
        'mwaskom__seaborn-2848',
        'pallets__flask-4045',
        'psf__requests-1963',
        'pydata__xarray-4094',
        'pylint-dev__pylint-5859',
        'pytest-dev__pytest-11143',
        'scikit-learn__scikit-learn-10297',
        'sphinx-doc__sphinx-8595'
    ]

    main(instance_ids_to_upload, force_regenerate_embeddings=args.force_regenerate_embeddings, include_chunk_content=args.include_chunk_content)