"""
multi-processing script to ingest the vector db.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import uuid
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
import requests
from datasets import load_dataset

from get_repo_structure.get_repo_structure import (
    checkout_commit,
    clone_repo,
    repo_to_top_folder,
)


def ingest_repo(args):
    bug, temp_playground_folder = args
    print(f"Ingesting repo for {bug['instance_id']}")

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(temp_playground_folder, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    repo_name, commit_id, instance_id = (
        bug["repo"],
        bug["base_commit"],
        bug["instance_id"],
    )
    clone_repo(repo_name, repo_playground)
    checkout_commit(f"{repo_playground}/{repo_to_top_folder[repo_name]}", commit_id)

    # ingest the repo into a vector db stored in the temp folder
    url = "http://127.0.0.1:8000/addRepo2/"

    headers = {
        "Content-Type": "application/json",
        "Accept": 'application/json, text/plain, /"',
        "Tenancy": "typefast/testing",
    }

    data = {
        "Name": repo_name,
        "Url": "https://github.com/" + repo_name + ".git",
        "IsPublic": False,
        "RootDir": os.path.abspath(repo_playground),
    }

    print("calling entelligence")
    print(data)
    response = requests.post(url, headers=headers, data=json.dumps(data))

    print(response.status_code)
    print(response.text)

    # clean up
    subprocess.run(
        ["rm", "-rf", f"{repo_playground}/{repo_to_top_folder[repo_name]}"], check=True
    )


if __name__ == "__main__":
    # use parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id", type=str, default="exploiter345/SWE-bench_Verified_50"
    )
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_top_n", type=int, default=1)
    parser.add_argument("--repo_filter", type=str, default="")
    parser.add_argument("--temp_playground_folder", type=str, default="playground")

    args = parser.parse_args()

    # load the dataset
    swe_bench_data = load_dataset(args.dataset_id, split=args.split_name)
    print("SWEBENCH dataset")
    print(swe_bench_data)

    if args.repo_filter != "":
        swe_bench_data = swe_bench_data.filter(lambda x: x["repo"] == args.repo_filter)

    if args.run_top_n > 0:
        swe_bench_data = swe_bench_data.select(range(args.run_top_n))

    # print the number of bugs in the dataset
    print(
        f"Number of instances in {args.dataset_id} {args.split_name} filtered by {args.repo_filter} is {len(swe_bench_data)}"
    )

    # do multiprocessing here to save the project structure for each bug
    with Pool(10) as p:
        p.map(
            ingest_repo, [(bug, args.temp_playground_folder) for bug in swe_bench_data]
        )
