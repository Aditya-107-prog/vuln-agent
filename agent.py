"""
agent.py - CLI entry point for the Vulnerability Finder Agent
Full pipeline: scan -> report (with approval) -> generate fixes ->
open a real PR (with approval).

Usage:
    python agent.py --target https://github.com/user/repo
    python agent.py --target /path/to/local/folder
    python agent.py --target . --output ./my-reports --verbose
"""

import argparse
import os
import sys
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in environment.")
    print("   1. Copy .env.example to .env")
    print("   2. Paste your Groq key from https://console.groq.com/keys")
    sys.exit(1)

from langgraph.types import Command
from agent.graph import build_graph
from logging_config import get_logger, bind_scan_id, clear_scan_id

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="Vulnerability Finder Agent -- scans Python repos, reports, fixes, and PRs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --target https://github.com/we45/Vulnerable-Flask-App
  python agent.py --target .
  python agent.py --target /path/to/project --output ./reports --verbose
        """
    )
    parser.add_argument("--target", "-t", required=True, help="GitHub URL or local folder path")
    parser.add_argument("--output", "-o", default="./reports", help="Directory to save the HTML report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output from each node")
    return parser.parse_args()


def print_status(step, total, message):
    bar = "X" * step + "." * (total - step)
    print(f"  [{bar}] {message}")


def print_report_approval_prompt(payload):
    print(f"\n{'-'*55}")
    print(f"  HUMAN APPROVAL REQUIRED -- Report Generation")
    print(f"{'-'*55}")
    print(f"  Repo            : {payload.get('repo_name')}")
    print(f"  Total findings  : {payload.get('total_findings')}")
    print(f"  Severity counts : {payload.get('severity_counts')}")
    print(f"{'-'*55}")
    answer = input("  Proceed to AI report generation? (y/n): ").strip().lower()
    return "approve" if answer in ("y", "yes") else "reject"


def print_pr_approval_prompt(payload):
    print(f"\n{'-'*55}")
    print(f"  HUMAN APPROVAL REQUIRED -- Pull Request")
    print(f"{'-'*55}")
    print(f"  Repo             : {payload.get('repo_name')}")
    code_fixes = payload.get("code_fixes", [])
    print(f"  Code fixes (confirmed) : {len(code_fixes)} file(s)")
    for fix in code_fixes:
        score = fix.get("critic_score")
        verdict = fix.get("critic_verdict")
        score_str = f"  [critic: {score}/10 {verdict}]" if score is not None else ""
        print(f"    - {fix['path']} ({fix['findings_addressed']} finding(s)){score_str}")

    dep_fixes = payload.get("dependency_fixes", [])
    print(f"  Dependency fixes       : {len(dep_fixes)} package(s)")
    for fix in dep_fixes:
        print(f"    - [{fix.get('type', '?').upper()}] {fix['package']}: {fix['old']} -> {fix['new']}")

    withheld_fixes = payload.get("withheld_code_fixes") or []
    if withheld_fixes:
        print(f"{'-'*55}")
        print(f"  \u26a0\ufe0f  {len(withheld_fixes)} fix(es) WITHHELD -- verification did NOT confirm these are resolved:")
        for fix in withheld_fixes:
            score = fix.get("critic_score")
            verdict = fix.get("critic_verdict")
            score_str = f"[critic: {score}/10 {verdict}]" if score is not None else "[unscored]"
            print(f"    - {fix['path']} ({fix['findings_addressed']} finding(s)) {score_str}")
            for problem in (fix.get("problems") or [])[:3]:
                print(f"        \u2022 {problem}")
        print(f"{'-'*55}")
        print(f"  These will be EXCLUDED from the PR by default.")
        print(f"  Type 'approve_all' (instead of 'y') if you want to include them anyway.")

    print(f"{'-'*55}")
    if withheld_fixes:
        answer = input("  Open a real GitHub PR? (y = confirmed fixes only / approve_all = include withheld / n): ").strip().lower()
        if answer in ("approve_all", "approve-all", "include_withheld", "all"):
            return "approve_all"
        return "approve" if answer in ("y", "yes") else "reject"
    else:
        answer = input("  Open a real GitHub PR with these fixes? (y/n): ").strip().lower()
        return "approve" if answer in ("y", "yes") else "reject"


def main():
    args = parse_args()

    if not args.verbose:
        os.environ["AGENT_QUIET"] = "1"

    os.makedirs(args.output, exist_ok=True)
    os.environ["AGENT_OUTPUT_DIR"] = os.path.abspath(args.output)

    print(f"\n{'-'*55}")
    print(f"  Vulnerability Finder Agent")
    print(f"{'-'*55}")
    print(f"  Target : {args.target}")
    print(f"  Output : {os.path.abspath(args.output)}")
    print(f"{'-'*55}\n")

    total_steps = 10  # router, fetch, code_scan, dep_scan, cve_enrich, human_review, report, fix_generate, pr_review, pr_generate
    start_time = time.time()
    print_status(0, total_steps, "Starting...")

    scan_id = str(uuid.uuid4())
    bind_scan_id(scan_id)
    logger.info(f"Scan started: target={args.target!r}")

    graph = build_graph()

    initial_state = {
        "target": args.target, "scan_id": scan_id, "input_type": None, "repo_path": None, "repo_name": None,
        "code_findings": None, "dep_findings": None, "enriched_findings": None,
        "human_decision": None, "report_markdown": None, "report_path": None,
        "code_fixes": None, "withheld_code_fixes": None, "dependency_fixes": None, "fixes_output_dir": None,
        "pr_decision": None, "pr_url": None, "error": None,
    }

    NODE_LABELS = {
        "router": "Detecting input type...",
        "github_fetch": "Cloning repository...",
        "local_read": "Reading local folder...",
        "code_scan": "Running static analysis (bandit)...",
        "dep_scan": "Scanning dependencies (pip-audit)...",
        "cve_enrich": "Enriching findings with CVE data...",
        "human_review": "Recording human decision...",
        "report": "Generating AI report...",
        "fix_generate": "Generating fixes (with critic review)...",
        "pr_review": "Recording PR decision...",
        "pr_generate": "Opening pull request...",
    }

    config = {"configurable": {"thread_id": scan_id}}

    step = 0
    final_state = None

    try:
        accumulated = {**initial_state}
        stream_input = initial_state

        while True:
            interrupted = False

            for event in graph.stream(stream_input, config, stream_mode="updates"):
                for node_name, node_output in event.items():

                    if node_name == "__interrupt__":
                        payload = node_output[0].value
                        checkpoint = payload.get("checkpoint")

                        if checkpoint == "report_approval":
                            answer = print_report_approval_prompt(payload)
                        elif checkpoint == "pr_approval":
                            answer = print_pr_approval_prompt(payload)
                        else:
                            print(f"\n  Unknown checkpoint, payload: {payload}")
                            answer = input("  Approve? (y/n): ").strip().lower()
                            answer = "approve" if answer in ("y", "yes") else "reject"

                        stream_input = Command(resume=answer)
                        interrupted = True
                        break

                    step += 1
                    label = NODE_LABELS.get(node_name, node_name)
                    print_status(step, total_steps, f"{node_name} done  --  {label}")
                    accumulated.update(node_output)

                if interrupted:
                    break

            if not interrupted:
                break

        final_state = accumulated

    except KeyboardInterrupt:
        print("\n\n  Scan interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Unhandled exception in CLI scan: {e}")
        print(f"\n\n  Unexpected error: {e}")
        if args.verbose:
            raise
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n{'-'*55}")

    if final_state and final_state.get("error"):
        print(f"  FAILED: {final_state['error']}")
        print(f"{'-'*55}\n")
        sys.exit(1)

    if final_state and final_state.get("human_decision") == "reject":
        print(f"  Scan stopped: report generation rejected by human reviewer.")
        print(f"{'-'*55}\n")
        sys.exit(0)

    if final_state:
        code_count  = len(final_state.get("code_findings") or [])
        dep_count   = len(final_state.get("dep_findings") or [])
        report_path = final_state.get("report_path", "")
        code_fixes  = final_state.get("code_fixes") or []
        withheld    = final_state.get("withheld_code_fixes") or []
        dep_fixes   = final_state.get("dependency_fixes") or []
        pr_url      = final_state.get("pr_url")

        print(f"  Scan complete in {elapsed:.1f}s")
        print(f"{'-'*55}")
        print(f"  Code issues found : {code_count}")
        print(f"  Dep vulns found   : {dep_count}")
        print(f"  Total findings    : {code_count + dep_count}")
        if report_path:
            print(f"{'-'*55}")
            print(f"  Report saved to:")
            print(f"  {report_path}")
        if code_fixes or dep_fixes or withheld:
            print(f"{'-'*55}")
            print(f"  Fixes generated   : {len(code_fixes)} confirmed code file(s), {len(dep_fixes)} dependency package(s)")
            if withheld:
                print(f"  Fixes withheld    : {len(withheld)} code file(s) NOT confirmed resolved (excluded from PR)")
        if final_state.get("pr_decision") == "reject":
            print(f"  PR generation rejected by human reviewer or skipped (no fixes / not a GitHub target).")
        if pr_url:
            print(f"{'-'*55}")
            print(f"  Pull request opened:")
            print(f"  {pr_url}")
        print(f"{'-'*55}\n")

    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()