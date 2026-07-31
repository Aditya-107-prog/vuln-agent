"""
agent/state.py
--------------
Defines VulnScanState -- the single shared memory object that flows
through every node in the graph.
"""

from typing import TypedDict, Optional


class VulnScanState(TypedDict):
    # --- Input ---
    target: str
    # scan_id MUST be declared here -- LangGraph builds its internal state
    # channels strictly from this TypedDict schema. Any key passed into
    # initial_state that isn't declared here (agent.py's and web_app.py's
    # initial_state dicts both set "scan_id") gets silently dropped as the
    # state enters the graph. That caused every node's
    # bind_scan_id(state.get("scan_id")) call to receive None and clobber
    # the correctly-bound value set by agent.py's main() / web_app.py's
    # run_scan_thread() before entering graph.stream() -- which is why
    # every agent.nodes log line showed scan_id "-" regardless of what was
    # bound upstream.
    scan_id: Optional[str]

    # --- After routing ---
    input_type: Optional[str]  # "github" or "local"

    # --- After fetching ---
    repo_path: Optional[str]
    repo_name: Optional[str]

    # --- After scanning ---
    code_findings: Optional[list]
    dep_findings: Optional[list]
    requirements_file_path: Optional[str]
    # go_mod_path / java_manifest_path: same class of bug as scan_id's
    # comment above describes -- these MUST be declared here or
    # LangGraph silently drops them between dep_scan_node and
    # fix_generate_node, even though dep_scan_node returns them
    # correctly. This exact omission caused Go/Java dependency
    # auto-fix to silently no-op (logging "No go.mod path provided" /
    # "No Java manifest path provided") despite dep_scan_node's own
    # logs confirming it found both paths moments earlier in the same
    # scan.
    go_mod_path: Optional[str]
    java_manifest_path: Optional[str]

    # --- After CVE enrichment ---
    enriched_findings: Optional[list]

    # --- After human review #1 (report generation checkpoint) ---
    human_decision: Optional[str]  # "approve" or "reject"

    # --- After report generation ---
    report_markdown: Optional[str]
    report_path: Optional[str]

    # --- After fix generation ---
    # code_fixes now holds ONLY fixes whose critique verdict was "pass"
    # (i.e. syntax valid AND bandit no longer flags the original issue).
    # withheld_code_fixes holds everything else (verdict "fail" or
    # "needs_improvement") -- these are NOT included in the PR by
    # default, since shipping a fix labeled "still broken" defeats the
    # point of having verified it in the first place.
    code_fixes: Optional[list]
    withheld_code_fixes: Optional[list]
    dependency_fixes: Optional[list]
    fixes_output_dir: Optional[str]

    # --- After human review #2 (PR checkpoint) ---
    pr_decision: Optional[str]  # "approve", "approve_all", or "reject"

    # --- After PR generation ---
    pr_url: Optional[str]

    # --- Error tracking ---
    error: Optional[str]