"""
agent/nodes.py -- all graph nodes, including PR generation with two
human-in-the-loop checkpoints (report approval, then PR approval).

VERIFICATION-GATED PR STEP (added after a real incident): previously,
fix_generate_node passed ALL generated fixes -- including ones whose
own critique verdict was "fail" (bandit still detects the original
vulnerability, or worse) -- straight into the PR step. The verdict was
only ever *displayed*, never *enforced*, so a fix labeled "still
broken" could still be shipped into a real PR just because the human
approved the batch as a whole without necessarily registering that one
line said "fail".

Now fix_generate_node splits code fixes into:
  - code_fixes: verdict == "pass" only -- these are what pr_review_node
    offers by default.
  - withheld_code_fixes: everything else -- excluded from the PR by
    default. Including one requires the human to type a distinct,
    explicit answer ("approve_all") at the PR checkpoint, not the same
    y/n used for the rest -- so shipping an unresolved fix can never
    happen by accident.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.types import interrupt

from tools.github_fetcher import fetch_github_repo
from tools.local_reader import read_local_directory
from tools.code_scanner import run_code_scan
from tools.dep_scanner import run_dependency_scan, run_go_dependency_scan, run_java_dependency_scan
from tools.cve_enricher import run_cve_enrichment
from tools.report_generator import generate_report, append_fix_results_to_report
from tools.fix_generator import generate_all_fixes
from tools.github_pr import (
    get_github_client, parse_owner_repo, fork_repo_if_needed,
    apply_fixes_and_push, open_pull_request
)
from agent.state import VulnScanState

from logging_config import get_logger, bind_scan_id

logger = get_logger(__name__)
QUIET = os.environ.get("AGENT_QUIET") == "1"

def _log(msg):
    logger.info(msg)


# ---------------------------------------------------------------------------
# Node 1: Router
# ---------------------------------------------------------------------------
def router_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    target = state["target"].strip()
    _log(f"\n[router] Target: {target}")

    if target.startswith("https://github.com/"):
        _log("[router] -> GitHub URL detected")
        return {"input_type": "github"}
    elif os.path.exists(target):
        _log("[router] -> Local path detected")
        return {"input_type": "local"}
    else:
        return {
            "input_type": None,
            "error": f"Cannot resolve target '{target}'. Provide a GitHub URL or a valid local path."
        }


def route_decision(state: VulnScanState) -> str:
    if state.get("error"):
        return "end"
    return "github_fetch" if state["input_type"] == "github" else "local_read"


# ---------------------------------------------------------------------------
# Node 2a: GitHub Fetcher
# ---------------------------------------------------------------------------
def github_fetch_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    result = fetch_github_repo(state["target"])
    if not result["success"]:
        error = result["error"]
        if "not found" in error.lower():
            error = (
                f"Repository not found: {state['target']}\n"
                "  - Check the URL is correct\n"
                "  - Private repos are not supported (no auth)"
            )
        return {"error": error}
    return {"repo_path": result["path"], "repo_name": result["repo_name"]}


# ---------------------------------------------------------------------------
# Node 2b: Local Reader
# ---------------------------------------------------------------------------
def local_read_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    target = state["target"]

    result = read_local_directory(target)
    if not result["success"]:
        return {"error": result["error"]}

    if result["file_count"] == 0:
        return {"error": f"Directory is empty: {target}"}

    if not result["python_files"]:
        _log("[local_read] No Python files found - bandit will have nothing to scan")

    repo_name = os.path.basename(result["path"].rstrip("/\\"))
    return {"repo_path": result["path"], "repo_name": repo_name}


# ---------------------------------------------------------------------------
# Node 3: Code Scanner
# ---------------------------------------------------------------------------
def code_scan_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    if state.get("error"):
        return {}

    result = run_code_scan(state["repo_path"])

    if not result["success"]:
        _log(f"[code_scan] Warning: {result['error']}")
        return {"code_findings": []}

    count = result["total"]
    _log(f"[code_scan] {count} code issue(s) found")
    return {"code_findings": result["findings"]}


# ---------------------------------------------------------------------------
# Node 4: Dependency Scanner
# ---------------------------------------------------------------------------
def dep_scan_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    if state.get("error"):
        return {}

    # requirements_file_path is used downstream by fix_generator.py to
    # edit requirements.txt for Python dependency fixes. go_mod_path and
    # java_manifest_path (captured below) serve the equivalent role for
    # Go/Java, added once fix generation was extended to cover those
    # ecosystems too.
    all_findings = []
    requirements_file_path = None

    # --- Python (pip-audit) -- unchanged from before this session ---
    py_result = run_dependency_scan(state["repo_path"])
    if not py_result["success"]:
        _log(f"[dep_scan] Warning (Python/pip-audit): {py_result['error']}")
    else:
        if not py_result["requirements_found"]:
            _log("[dep_scan] No requirements file found, skipping Python dependency scan")
        all_findings.extend(py_result["findings"])
        requirements_file_path = py_result.get("requirements_file")

    # --- Go (govulncheck) ---
    go_mod_path = None
    go_result = run_go_dependency_scan(state["repo_path"])
    if not go_result["success"]:
        # Distinguish "not installed" from a real scan failure in the
        # log, since the former is an environment setup issue the
        # operator needs to fix, not a bug -- but either way this must
        # NOT be fatal to the rest of the scan (same non-fatal-warning
        # pattern the Python path already uses above).
        _log(f"[dep_scan] Warning (Go/govulncheck): {go_result['error']}")
    else:
        if go_result["requirements_found"]:
            _log(f"[dep_scan] Go module found ({go_result['requirements_file']}) -- {go_result['total']} vulnerable package(s)")
        all_findings.extend(go_result["findings"])
        go_mod_path = go_result.get("requirements_file")

    # --- Java (osv-scanner) ---
    java_manifest_path = None
    java_result = run_java_dependency_scan(state["repo_path"])
    if not java_result["success"]:
        _log(f"[dep_scan] Warning (Java/osv-scanner): {java_result['error']}")
    else:
        if java_result["requirements_found"]:
            _log(f"[dep_scan] Java manifest found ({java_result['requirements_file']}) -- {java_result['total']} vulnerable package(s)")
        all_findings.extend(java_result["findings"])
        java_manifest_path = java_result.get("requirements_file")

    _log(f"[dep_scan] {len(all_findings)} vulnerable package(s) found total (Python + Go + Java)")
    return {
        "dep_findings": all_findings,
        "requirements_file_path": requirements_file_path,
        "go_mod_path": go_mod_path,
        "java_manifest_path": java_manifest_path,
    }


# ---------------------------------------------------------------------------
# Node 5: CVE Enricher
# ---------------------------------------------------------------------------
def cve_enrich_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    if state.get("error"):
        return {}

    code_findings = state.get("code_findings") or []
    dep_findings  = state.get("dep_findings") or []

    if not code_findings and not dep_findings:
        _log("[cve_enricher] No findings to enrich")
        return {"enriched_findings": []}

    try:
        enriched = run_cve_enrichment(code_findings, dep_findings)
    except Exception as e:
        logger.warning(f"CVE enrichment failed ({e}), using raw findings", exc_info=True)
        enriched = [
            {**f, "cvss_score": None, "severity": f.get("severity", "MEDIUM"),
             "osv_ids": [], "finding_type": "code"}
            for f in code_findings
        ] + [
            {**f, "cvss_score": None, "severity": "HIGH",
             "osv_ids": [], "finding_type": "dependency"}
            for f in dep_findings
        ]

    return {"enriched_findings": enriched}


# ---------------------------------------------------------------------------
# Node 6: Human Review #1 -- report generation checkpoint
# ---------------------------------------------------------------------------
def human_review_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    if state.get("error"):
        return {}

    enriched = state.get("enriched_findings") or []
    severity_counts = {}
    for f in enriched:
        sev = f.get("severity", "UNKNOWN")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    decision = interrupt({
        "checkpoint": "report_approval",
        "question": "Proceed to AI report generation for these findings?",
        "repo_name": state.get("repo_name"),
        "total_findings": len(enriched),
        "severity_counts": severity_counts,
    })

    normalized = str(decision).strip().lower()
    approved = normalized in ("approve", "approved", "yes", "y")

    _log(f"[human_review] Decision received: {decision!r} -> {'APPROVED' if approved else 'REJECTED'}")

    return {"human_decision": "approve" if approved else "reject"}


def human_review_decision(state: VulnScanState) -> str:
    if state.get("error"):
        return "end"
    return "report" if state.get("human_decision") == "approve" else "end"


# ---------------------------------------------------------------------------
# Node 7: Report Generator
# ---------------------------------------------------------------------------
def report_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    if state.get("error"):
        _log(f"\n[report] Skipping - error upstream: {state['error']}")
        return {}

    repo_name         = state.get("repo_name", "unknown-repo")
    enriched_findings = state.get("enriched_findings") or []

    output_dir = os.environ.get("AGENT_OUTPUT_DIR")

    result = generate_report(repo_name, enriched_findings, output_dir=output_dir)

    if not result["success"]:
        _log(f"[report] Report generation failed: {result.get('error')}")
        return {"error": result.get("error")}

    return {
        "report_markdown": result["report_markdown"],
        "report_path":     result["report_path"]
    }


# ---------------------------------------------------------------------------
# Node 8: Fix Generator (code fixes + dependency fixes, critic-reviewed)
# ---------------------------------------------------------------------------
def fix_generate_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    if state.get("error"):
        return {}

    enriched = state.get("enriched_findings") or []
    if not enriched:
        _log("[fix_generate] No findings to fix")
        return {"code_fixes": [], "withheld_code_fixes": [], "dependency_fixes": []}

    try:
        result = generate_all_fixes(
            state["repo_path"], state["repo_name"], enriched,
            output_dir="./proposed_fixes",
            requirements_file_path=state.get("requirements_file_path"),
            go_mod_path=state.get("go_mod_path"),
            java_manifest_path=state.get("java_manifest_path"),
            scan_id=state.get("scan_id"),
        )
    except Exception as e:
        logger.error(f"Fix generation failed: {e}", exc_info=True)
        return {"code_fixes": [], "withheld_code_fixes": [], "dependency_fixes": []}

    all_code_fixes = result["code_fixes"]

    # --- Gate: only fixes with a "pass" verdict are eligible for the PR
    # by default. A "fail" or "needs_improvement" verdict means bandit
    # still detects the original issue, or the critic found a critical
    # problem -- shipping that silently would defeat the point of
    # verifying it in the first place.
    confirmed_fixes = []
    withheld_fixes = []
    for fix in all_code_fixes:
        verdict = (fix.get("critique") or {}).get("verdict")
        if verdict == "pass":
            confirmed_fixes.append(fix)
        else:
            withheld_fixes.append(fix)

    if withheld_fixes:
        _log(f"[fix_generate] {len(withheld_fixes)} fix(es) WITHHELD from PR by default (verdict != pass):")
        for f in withheld_fixes:
            verdict = (f.get("critique") or {}).get("verdict", "unknown")
            _log(f"[fix_generate]   - {f['relative_path']} (verdict: {verdict})")

    # Append fix verification results (critic scores, problems, per-layer
    # verification detail) to the already-generated report. This can only
    # happen here, not during report_node -- fix data doesn't exist yet at
    # that point in the pipeline (report generation is human-approval
    # checkpoint #1, before any fixes are attempted). Failure here is
    # logged but non-fatal -- it must never block the rest of the
    # pipeline (PR review, PR generation) over a report-formatting issue.
    report_path = state.get("report_path")
    updated_report_markdown = state.get("report_markdown")
    if report_path:
        append_result = append_fix_results_to_report(
            report_path=report_path,
            report_markdown=state.get("report_markdown"),
            repo_name=state.get("repo_name", "unknown-repo"),
            code_fixes=confirmed_fixes,
            withheld_fixes=withheld_fixes,
            dependency_fixes=result["dependency_fixes"],
        )
        if append_result["success"]:
            _log(f"[fix_generate] Appended fix results to report: {report_path}")
            updated_report_markdown = append_result["report_markdown"]
        else:
            _log(f"[fix_generate] Warning: could not append fix results to report: {append_result.get('error')}")

    return {
        "code_fixes": confirmed_fixes,
        "withheld_code_fixes": withheld_fixes,
        "dependency_fixes": result["dependency_fixes"],
        "fixes_output_dir": result["output_dir"],
        "report_markdown": updated_report_markdown,
    }


# ---------------------------------------------------------------------------
# Node 9: Human Review #2 -- PR checkpoint
# ---------------------------------------------------------------------------
def pr_review_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    if state.get("error"):
        return {}

    code_fixes = state.get("code_fixes") or []
    withheld_fixes = state.get("withheld_code_fixes") or []
    dependency_fixes = state.get("dependency_fixes") or []

    if not code_fixes and not withheld_fixes and not dependency_fixes:
        _log("[pr_review] No fixes were generated -- nothing to PR")
        return {"pr_decision": "reject"}

    if state.get("input_type") != "github":
        _log("[pr_review] Target is a local folder, not a GitHub repo -- skipping PR step")
        return {"pr_decision": "reject"}

    payload = {
        "checkpoint": "pr_approval",
        "question": "Open a real GitHub PR with these fixes?",
        "repo_name": state.get("repo_name"),
        "code_fixes": [
            {
                "path": f["relative_path"],
                "findings_addressed": f["findings_addressed"],
                "critic_score": (f.get("critique") or {}).get("score"),
                "critic_verdict": (f.get("critique") or {}).get("verdict"),
                "verification": f.get("verification"),
            }
            for f in code_fixes
        ],
        "dependency_fixes": [
            {"package": f["package"], "old": f["old_line"], "new": f["new_line"], "type": f.get("fix_type")}
            for f in dependency_fixes
        ],
    }

    if withheld_fixes:
        payload["withheld_code_fixes"] = [
            {
                "path": f["relative_path"],
                "findings_addressed": f["findings_addressed"],
                "critic_score": (f.get("critique") or {}).get("score"),
                "critic_verdict": (f.get("critique") or {}).get("verdict"),
                "verification": f.get("verification"),
                "problems": [
                    p.get("text") if isinstance(p, dict) else p
                    for p in (f.get("critique") or {}).get("problems", [])
                ],
            }
            for f in withheld_fixes
        ]
        payload["withheld_notice"] = (
            f"{len(withheld_fixes)} fix(es) are NOT included above because verification "
            f"did not confirm they resolve the vulnerability (bandit still flags the "
            f"original issue, or the critic found a critical problem). They will be "
            f"excluded from the PR unless you explicitly type 'approve_all' instead of 'y'."
        )
        _log(f"\n[pr_review] NOTE: {payload['withheld_notice']}")
        for f in payload["withheld_code_fixes"]:
            _log(f"[pr_review]   - {f['path']} (verdict: {f['critic_verdict']})")

    decision = interrupt(payload)
    normalized = str(decision).strip().lower()

    include_withheld = normalized in ("approve_all", "approve-all", "include_withheld")
    approved = include_withheld or normalized in ("approve", "approved", "yes", "y")

    if include_withheld and withheld_fixes:
        _log(f"[pr_review] Decision received: {decision!r} -> APPROVED, INCLUDING {len(withheld_fixes)} withheld (unconfirmed) fix(es) -- explicit override")
        code_fixes = code_fixes + withheld_fixes
    else:
        _log(f"[pr_review] Decision received: {decision!r} -> {'APPROVED (confirmed fixes only)' if approved else 'REJECTED'}")
        if withheld_fixes and approved:
            _log(f"[pr_review] {len(withheld_fixes)} unconfirmed fix(es) excluded from this PR. Re-run with 'approve_all' to include them anyway.")

    return {
        "pr_decision": "approve" if approved else "reject",
        "code_fixes": code_fixes,
    }


def pr_review_decision(state: VulnScanState) -> str:
    if state.get("error"):
        return "end"
    return "pr_generate" if state.get("pr_decision") == "approve" else "end"


# ---------------------------------------------------------------------------
# Node 10: PR Generator -- fork/branch/commit/push/PR
# ---------------------------------------------------------------------------
def pr_generate_node(state: VulnScanState) -> dict:
    bind_scan_id(state.get("scan_id"))
    if state.get("error"):
        return {}

    github_username = os.environ.get("GITHUB_USERNAME")
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_username or not github_token:
        return {"error": "GITHUB_USERNAME / GITHUB_TOKEN not set in environment"}

    code_fixes = state.get("code_fixes") or []
    if not code_fixes and not (state.get("dependency_fixes") or []):
        _log("[pr_generate] No confirmed fixes to include -- skipping PR generation")
        return {"error": None, "pr_url": None}

    try:
        owner, repo_name = parse_owner_repo(state["target"])
        gh = get_github_client()
        fork_full_name, did_fork = fork_repo_if_needed(gh, owner, repo_name)
        fork_owner = fork_full_name.split("/")[0]
        branch_name = f"vuln-agent-fixes-{int(time.time())}"

        apply_fixes_and_push(
            fork_full_name=fork_full_name,
            branch_name=branch_name,
            code_fixes=code_fixes,
            dependency_fixes=state.get("dependency_fixes") or [],
            github_username=github_username,
            github_token=github_token,
            workdir="./pr_workdir",
        )

        pr_url = open_pull_request(
            gh, owner, repo_name, fork_owner, branch_name,
            code_fixes, state.get("dependency_fixes") or [],
            same_repo=not did_fork,
        )
        _log(f"[pr_generate] Pull request opened: {pr_url}")
        return {"pr_url": pr_url}

    except Exception as e:
        logger.exception(f"PR generation failed for {state.get('repo_name')}")
        return {"error": f"PR generation failed: {e}"}