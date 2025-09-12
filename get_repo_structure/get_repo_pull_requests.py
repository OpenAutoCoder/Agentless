import requests
import json
import time
import os

TOKEN = "TOKEN"
HEADERS = {"Authorization": f"token {TOKEN}"}

# Languages to include
LANGUAGES = ["Go", "Java", "JavaScript", "Python"]

# How many repos per language to fetch
TOP_N_REPOS = 10

# Max PRs per repo
MAX_PRS = 100
OUTPUT_FILE = "pull_requests_dataset.json"


def safe_get(d, *keys, default=None):
    """Safely get nested dict values"""
    for key in keys:
        if d is None or key not in d:
            return default
        d = d[key]
    return d


def get_top_repos(language, top_n=TOP_N_REPOS):
    """Fetch top repositories by stars for a given language"""
    url = f"https://api.github.com/search/repositories?q=language:{language}&sort=stars&order=desc&per_page={top_n}"
    r = requests.get(url, headers=HEADERS)
    repos = []
    if r.status_code == 200:
        for item in r.json()["items"]:
            repos.append({
                "name": item["full_name"],
                "language": item["language"],
                "stars": item["stargazers_count"],
                "license": safe_get(item, "license", "spdx_id"),
                "url": item["html_url"]
            })
    else:
        print(f"Error fetching repos for {language}: {r.status_code}")
    return repos


def get_pull_requests(repo_full_name, max_prs=MAX_PRS):
    """Fetch latest PRs for a repository"""
    prs = []
    url = f"https://api.github.com/repos/{repo_full_name}/pulls?state=closed&per_page=100"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"Error fetching PRs for {repo_full_name}: {r.status_code}")
        return prs

    pr_list = r.json()[:max_prs]
    for pr in pr_list:
        pr_data = {
            "id": pr.get("number"),
            "title": pr.get("title"),
            "body": pr.get("body"),
            "author": safe_get(pr, "user", "login", default="ghost"),
            "state": pr.get("state"),
            "created_at": pr.get("created_at"),
            "merged_at": pr.get("merged_at"),
            "url": pr.get("html_url"),
            "commits": [],
            "comments": []
        }

        # Commits
        commits_r = requests.get(pr.get("commits_url"), headers=HEADERS)
        if commits_r.status_code == 200:
            for commit in commits_r.json():
                pr_data["commits"].append({
                    "sha": commit.get("sha"),
                    "message": safe_get(commit, "commit", "message"),
                    "author": safe_get(commit, "commit", "author", "name", default="unknown"),
                    "url": commit.get("html_url")
                })

        # Review comments
        comments_url = safe_get(pr, "_links", "review_comments", "href")
        if comments_url:
            comments_r = requests.get(comments_url, headers=HEADERS)
            if comments_r.status_code == 200:
                for comment in comments_r.json():
                    pr_data["comments"].append({
                        "user": safe_get(comment, "user", "login", default="ghost"),
                        "body": comment.get("body"),
                        "created_at": comment.get("created_at"),
                        "url": comment.get("html_url"),
                        "line": comment.get("line"),
                        "position": comment.get("position"),
                        "commit_id": comment.get("commit_id"),
                        "type": "inline" if comment.get("path") else "general"
                    })

        prs.append(pr_data)
        time.sleep(0.5)  # avoid spamming API
    return prs


def build_dataset(output_file=OUTPUT_FILE):
    """Build dataset of PRs from top repos with incremental saving"""
    # Load existing dataset if present
    dataset = []
    seen_repos = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        seen_repos = {repo["repo_name"] for repo in dataset}

    for lang in LANGUAGES:
        print(f"Fetching top repos for {lang} ...")
        top_repos = get_top_repos(lang)
        for repo in top_repos:
            # Check if repo is already in dataset
            existing_repo = next((r for r in dataset if r["repo_name"] == repo["name"]), None)

            # Skip if it already has PRs with comments
            if existing_repo and existing_repo.get("pull_requests"):
                has_comments = any(pr["comments"] for pr in existing_repo["pull_requests"])
                if has_comments:
                    print(f"Skipping {repo['name']} (already has PRs with comments)")
                    continue
                else:
                    print(f"Refetching {repo['name']} (previously empty or no comments)")

            print(f"Fetching PRs for {repo['name']} ...")
            prs = get_pull_requests(repo["name"])
            repo_entry = {
                "repo_name": repo["name"],
                "language": repo["language"],
                "stars": repo["stars"],
                "license": repo["license"],
                "url": repo["url"],
                "pull_requests": prs
            }

            if existing_repo:
                dataset[dataset.index(existing_repo)] = repo_entry
            else:
                dataset.append(repo_entry)

            # Save progress immediately
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset with {len(dataset)} repos to {output_file}")


def fetch_specific_repos(repo_list, output_file=OUTPUT_FILE):
    """Fetch PRs for a list of specific repositories and update the dataset incrementally"""
    # Load existing dataset if present
    dataset = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    for repo_full_name in repo_list:
        existing_repo = next((r for r in dataset if r["repo_name"] == repo_full_name), None)

        # Skip if already has PRs with comments
        if existing_repo and any(pr["comments"] for pr in existing_repo.get("pull_requests", [])):
            print(f"Skipping {repo_full_name} (already has PRs with comments)")
            continue
        else:
            print(f"Refetching {repo_full_name} (previously empty or no comments)")

        print(f"Fetching PRs for {repo_full_name} ...")
        prs = get_pull_requests(repo_full_name)

        # Fetch repo info via GitHub API
        r = requests.get(f"https://api.github.com/repos/{repo_full_name}", headers=HEADERS)
        if r.status_code != 200:
            print(f"Error fetching repo info for {repo_full_name}: {r.status_code}")
            continue
        repo_data = r.json()

        repo_entry = {
            "repo_name": repo_full_name,
            "language": repo_data.get("language"),
            "stars": repo_data.get("stargazers_count"),
            "license": safe_get(repo_data, "license", "spdx_id"),
            "url": repo_data.get("html_url"),
            "pull_requests": prs
        }

        if existing_repo:
            dataset[dataset.index(existing_repo)] = repo_entry
        else:
            dataset.append(repo_entry)

        # Save progress immediately
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Updated dataset with {len(dataset)} repos to {output_file}")


# ----------------------------
# Run main dataset builder
# ----------------------------
build_dataset()

# ----------------------------
# Fetch specific Atlassian repos
# ----------------------------
SPECIFIC_REPOS = [
    "atlassian/atlascode",
    "atlassian/dc-app-performance-toolkit"
]
fetch_specific_repos(SPECIFIC_REPOS)
