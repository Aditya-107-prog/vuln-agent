"""
tools/github_pr.py
-------------------
Step 2 of PR generation: forks the target repo, applies the fixes
already generated (and already human-reviewed on disk) by
fix_generator.py onto a new branch, pushes to the fork, and opens a
real pull request against the original repo.

Nothing in this file runs without explicit human approval -- see
test_pr_generator.py for the approval checkpoint.
"""

import os
import shutil
import stat
import subprocess
import time
from github import Github, Auth

from logging_config import get_logger

logger = get_logger(__name__)


def _force_remove_readonly(func, path, exc_info):
    """shutil.rmtree error handler for Windows: git marks files inside
    .git/objects/ as read-only, which makes os.unlink() fail with
    PermissionError on Windows (this isn't an issue on Mac/Linux). Force
    the file writable, then retry the operation that failed."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree_safe(path):
    shutil.rmtree(path, onexc=_force_remove_readonly)


def _run_git(args, cwd, check=True):
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def get_github_client() -> Github:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set in environment")
    return Github(auth=Auth.Token(token))


def parse_owner_repo(github_url: str) -> tuple:
    """Extract (owner, repo) from a GitHub URL like https://github.com/owner/repo"""
    clean = github_url.rstrip("/").replace(".git", "")
    parts = clean.split("/")
    return parts[-2], parts[-1]


def fork_repo_if_needed(gh: Github, owner: str, repo_name: str, wait_seconds: int = 20) -> tuple:
    """Returns (fork_full_name, needed_fork: bool).

    GitHub does NOT allow forking a repo into the same account that
    already owns it -- attempting this returns a confusing 403
    ("Resource not accessible by personal access token"), not a clear
    "you already own this" error. If the authenticated user already
    owns the repo, they already have write access, so we skip forking
    entirely and just push directly to a new branch on the same repo.
    """
    user = gh.get_user()

    if user.login.lower() == owner.lower():
        logger.info(f"You already own {owner}/{repo_name} -- skipping fork, pushing directly.")
        return f"{owner}/{repo_name}", False

    upstream = gh.get_repo(f"{owner}/{repo_name}")
    logger.info(f"Forking {owner}/{repo_name} -> {user.login}/{repo_name} ...")
    user.create_fork(upstream)

    fork_full_name = f"{user.login}/{repo_name}"
    for _ in range(wait_seconds):
        try:
            gh.get_repo(fork_full_name)
            logger.info(f"Fork ready: {fork_full_name}")
            return fork_full_name, True
        except Exception:
            time.sleep(1)

    raise RuntimeError(f"Fork did not become ready after {wait_seconds}s: {fork_full_name}")


def apply_fixes_and_push(
    fork_full_name: str,
    branch_name: str,
    code_fixes: list,
    dependency_fixes: list,
    github_username: str,
    github_token: str,
    workdir: str,
) -> str:
    """Clones the fork fresh, creates a branch, applies the already-generated
    fix files onto their matching paths, commits, and pushes."""

    os.makedirs(workdir, exist_ok=True)
    clone_dir = os.path.join(workdir, "pr_clone")
    if os.path.exists(clone_dir):
        _rmtree_safe(clone_dir)

    auth_url = f"https://{github_username}:{github_token}@github.com/{fork_full_name}.git"
    logger.info(f"Cloning fork {fork_full_name} ...")
    # NOTE: clone_dir already includes the workdir prefix (e.g.
    # "./pr_workdir/pr_clone"), so this must run from the project root
    # (cwd=None), NOT cwd=workdir -- otherwise git resolves clone_dir
    # relative to workdir too, producing a doubled/nested path.
    _run_git(["clone", auth_url, clone_dir], cwd=None)

    _run_git(["checkout", "-b", branch_name], cwd=clone_dir)

    changed_files = []

    for fix in code_fixes:
        dest_path = os.path.join(clone_dir, fix["relative_path"])
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(fix["fixed_path"], dest_path)
        changed_files.append(fix["relative_path"])
        logger.info(f"Applied fix: {fix['relative_path']}")

    if dependency_fixes:
        req_relative_path = dependency_fixes[0].get("requirements_relative_path", "requirements.txt")
        req_dest = os.path.join(clone_dir, req_relative_path)
        dest_dir = os.path.dirname(req_dest)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(dependency_fixes[0]["fixed_path"], req_dest)
        changed_files.append(req_relative_path)
        logger.info(f"Applied fix: {req_relative_path}")

    if not changed_files:
        raise RuntimeError("No fixed files to apply -- nothing to commit")

    _run_git(["add"] + changed_files, cwd=clone_dir)
    _run_git(
        ["-c", "user.email=vuln-agent@example.com", "-c", "user.name=Vulnerability Finder Agent",
         "commit", "-m", "fix: automated security fixes from Vulnerability Finder Agent"],
        cwd=clone_dir
    )
    _run_git(["push", "-u", "origin", branch_name], cwd=clone_dir)

    logger.info(f"Pushed branch '{branch_name}' to {fork_full_name}")
    return branch_name


def open_pull_request(
    gh: Github, owner: str, repo_name: str, fork_owner: str, branch_name: str,
    code_fixes: list, dependency_fixes: list, same_repo: bool
) -> str:
    """Opens a PR for branch_name against owner/repo_name's default branch.
    If same_repo is True, head is just the branch name (no owner prefix).
    If it's a genuine fork, head must be "fork_owner:branch_name".
    Returns the PR URL."""
    upstream = gh.get_repo(f"{owner}/{repo_name}")
    default_branch = upstream.default_branch

    head = branch_name if same_repo else f"{fork_owner}:{branch_name}"

    title = f"Security fixes: {len(code_fixes)} code issue(s), {len(dependency_fixes)} dependency issue(s)"

    body_lines = ["## Automated Security Fixes", "", "Generated by Vulnerability Finder Agent.", ""]
    if code_fixes:
        body_lines.append("### Code fixes")
        for fix in code_fixes:
            body_lines.append(f"- `{fix['relative_path']}`: {fix['explanation']}")
            critique = fix.get("critique")
            if critique and critique.get("verdict") not in (None, "unavailable"):
                body_lines.append(f"  - Independent review (Mistral/Bedrock): {critique.get('score')}/10 ({critique.get('verdict')})")
    if dependency_fixes:
        body_lines.append("")
        body_lines.append("### Dependency fixes")
        for fix in dependency_fixes:
            tag = "direct bump" if fix.get("fix_type") == "direct" else "transitive dependency, new pin added"
            body_lines.append(f"- `{fix['package']}` ({tag}): `{fix['old_line']}` -> `{fix['new_line']}`")
    body_lines.append("")
    body_lines.append("**Please review carefully before merging.** This PR was generated by an AI agent and has not been human-authored.")

    # Right after a fork is created and pushed to, GitHub's internal
    # fork-relationship cache can lag a few seconds before create_pull
    # recognizes it -- even though clone/push already worked. Retry on
    # 404 rather than failing immediately.
    from github import GithubException

    last_error = None
    for attempt in range(6):
        try:
            pr = upstream.create_pull(
                title=title,
                body="\n".join(body_lines),
                head=head,
                base=default_branch,
            )
            return pr.html_url
        except GithubException as e:
            if e.status == 404 and attempt < 5:
                logger.warning(f"PR creation got 404 (fork propagation delay), retrying in 3s... (attempt {attempt + 1}/6)")
                time.sleep(3)
                last_error = e
                continue
            raise

    raise last_error