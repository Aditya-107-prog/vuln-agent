"""
Tool 1: GitHub Fetcher
-----------------------
Accepts a public GitHub repo URL, clones it to a temp directory,
and returns the local path to the cloned repo.

Handles:
- Malformed URLs
- Non-GitHub URLs
- Repos that don't exist (git clone fails)
"""

import subprocess
import tempfile
import os
import re

from logging_config import get_logger

logger = get_logger(__name__)


def fetch_github_repo(url: str) -> dict:
    """
    Clone a public GitHub repo to a temp directory.

    Args:
        url: A GitHub repo URL e.g. https://github.com/user/repo

    Returns:
        {
            "success": True,
            "path": "/tmp/xyz/repo",   # local path to cloned repo
            "repo_name": "repo",
            "error": None
        }
        or on failure:
        {
            "success": False,
            "path": None,
            "repo_name": None,
            "error": "description of what went wrong"
        }
    """

    # --- Validate it looks like a GitHub URL ---
    # We expect: https://github.com/username/reponame
    pattern = r"^https://github\.com/[\w\-\.]+/[\w\-\.]+$"
    url = url.strip().rstrip("/")  # clean up any trailing slash

    if not re.match(pattern, url):
        return {
            "success": False,
            "path": None,
            "repo_name": None,
            "error": f"Invalid GitHub URL: '{url}'. Expected format: https://github.com/user/repo"
        }

    # Extract repo name from URL (last segment)
    repo_name = url.split("/")[-1]

    # --- Create a temp directory to clone into ---
    # tempfile.mkdtemp() creates a real folder on disk that persists
    # until we explicitly delete it (or the OS cleans up on reboot)
    temp_dir = tempfile.mkdtemp(prefix="vuln_agent_")

    clone_path = os.path.join(temp_dir, repo_name)

    logger.info(f"Cloning {url} -> {clone_path}")

    # --- Run git clone as a subprocess ---
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, clone_path],
            # --depth 1 = shallow clone, only latest commit (much faster)
            capture_output=True,   # capture stdout and stderr
            text=True,             # decode bytes to string automatically
            timeout=60             # bail if it takes more than 60s
        )

        if result.returncode != 0:
            # git clone failed — repo probably doesn't exist or is private
            error = f"git clone failed: {result.stderr.strip()}"
            logger.error(error)
            return {
                "success": False,
                "path": None,
                "repo_name": repo_name,
                "error": error
            }

        logger.info(f"Cloned successfully to {clone_path}")
        return {
            "success": True,
            "path": clone_path,
            "repo_name": repo_name,
            "error": None
        }

    except FileNotFoundError:
        # git is not installed on this machine
        logger.error("git is not installed or not in PATH")
        return {
            "success": False,
            "path": None,
            "repo_name": repo_name,
            "error": "git is not installed or not in PATH"
        }

    except subprocess.TimeoutExpired:
        logger.error(f"git clone timed out after 60 seconds ({url})")
        return {
            "success": False,
            "path": None,
            "repo_name": repo_name,
            "error": "git clone timed out after 60 seconds"
        }


# --- Quick sanity check ---
if __name__ == "__main__":
    # Test 1: valid public repo
    print("=== Test 1: Valid repo ===")
    result = fetch_github_repo("https://github.com/anshumanbhardwaj/flask-vulnerable-app")
    print(result)

    # Test 2: malformed URL
    print("\n=== Test 2: Bad URL ===")
    result = fetch_github_repo("not-a-url")
    print(result)

    # Test 3: repo that doesn't exist
    print("\n=== Test 3: Non-existent repo ===")
    result = fetch_github_repo("https://github.com/fakeuser99999/fakerepo88888")
    print(result)