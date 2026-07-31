"""
Tool: Report Generator
Calls Groq (Llama 3) to write a Markdown report from enriched findings,
converts to HTML, saves with timestamp.

LangFuse instrumentation: generate_report() is the top-level trace.
The actual Groq LLM call is logged as a child "generation" observation
inside it, capturing the prompt, the raw output, and token usage.
Groq isn't natively wrapped by Langfuse (unlike OpenAI), so this is
done manually via start_as_current_observation().
"""

import os
import json
from datetime import datetime
from groq import Groq
import markdown as md_lib
from langfuse import observe, get_client

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")

LANGFUSE_ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


# Marker spliced into the initial report between "Findings Breakdown" and
# "Recommendations". append_fix_results_to_report() later replaces it with
# the Verification Methodology + Fix Results section once fixes have been
# attempted. If fixes are never attempted (or that step fails), the marker
# is an HTML comment so it stays invisible in the rendered report rather
# than leaving a broken heading.
FIX_RESULTS_MARKER = "<!-- FIX_RESULTS_PLACEHOLDER -->"

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def _severity_counts(enriched_findings: list) -> dict:
    counts = {s: 0 for s in SEVERITY_ORDER}
    for f in enriched_findings:
        sev = str(f.get("severity", "")).upper()
        if sev not in counts:
            counts[sev] = 0
        counts[sev] += 1
    return counts


def _md_escape_cell(value) -> str:
    """Keeps a single table row from breaking if a value contains a pipe
    or newline (e.g. a code snippet or multi-line CVE description)."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _shorten_file_path(file_path: str, repo_name: str) -> str:
    """Findings carry the absolute path to the cloned repo inside our
    temp sandbox (e.g. C:\\Users\\...\\vuln_agent_xxxx\\repo_name\\app.py),
    which is both unreadable and, being one long unbroken string with no
    spaces, prone to overflowing table columns even with wrapping CSS.
    Trims everything up to and including the last occurrence of the repo
    name, leaving a clean relative path like `app.py` or `pkg/module.py`.
    Falls back to the original string unchanged if the repo name isn't
    found in it (e.g. paths already relative)."""
    if not file_path or not repo_name:
        return file_path
    normalized = file_path.replace("\\", "/")
    marker = f"/{repo_name}/"
    idx = normalized.rfind(marker)
    if idx == -1:
        return file_path
    return normalized[idx + len(marker):]


def build_findings_breakdown_markdown(enriched_findings: list, repo_name: str = "") -> str:
    """Builds the '## Findings Breakdown' section directly from
    enriched_findings -- no LLM involved. Deterministic so the same scan
    always produces the same table, and so we never risk the model
    dropping, inventing, or mis-transcribing a finding.
    """
    code_findings = [f for f in enriched_findings if f.get("finding_type") == "code"]
    dep_findings = [f for f in enriched_findings if f.get("finding_type") == "dependency"]

    lines = ["## Findings Breakdown\n"]

    lines.append("### Code Issues\n")
    if not code_findings:
        lines.append("No code issues were found.\n")
    else:
        lines.append("| File | Line | Severity | Rule ID | Description |")
        lines.append("|---|---|---|---|---|")
        for f in code_findings:
            rule_id = f.get("issue") or "N/A"
            lines.append(
                "| {file} | {line} | {sev} | {rule} | {desc} |".format(
                    file=_md_escape_cell(_shorten_file_path(f.get("file", ""), repo_name)),
                    line=_md_escape_cell(f.get("line", "")),
                    sev=_md_escape_cell(f.get("severity", "")),
                    rule=_md_escape_cell(rule_id),
                    desc=_md_escape_cell(f.get("description", ""))[:200],
                )
            )
        lines.append("")

    lines.append("### Dependency Vulnerabilities\n")
    if not dep_findings:
        lines.append("No dependency vulnerabilities were found.\n")
    else:
        lines.append("| Package | Installed Version | Fix Version | Severity | CVE ID |")
        lines.append("|---|---|---|---|---|")
        for f in dep_findings:
            osv_ids = f.get("osv_ids") or []
            cve_id = ", ".join(osv_ids) if osv_ids else "N/A"
            lines.append(
                "| {pkg} | {installed} | {fix} | {sev} | {cve} |".format(
                    pkg=_md_escape_cell(f.get("package", "")),
                    installed=_md_escape_cell(f.get("installed_version", "")),
                    fix=_md_escape_cell(f.get("fix_version", "no fix available")),
                    sev=_md_escape_cell(f.get("severity", "")),
                    cve=_md_escape_cell(cve_id),
                )
            )
        lines.append("")

    return "\n".join(lines)


def build_scan_summary_markdown(enriched_findings: list) -> str:
    """Builds the '## Scan Summary' severity-count table directly from
    enriched_findings -- deterministic, same reasoning as the findings
    breakdown above.
    """
    counts = _severity_counts(enriched_findings)
    total = sum(counts.values())

    lines = ["## Scan Summary\n", "| Category | Count |", "|---|---|"]
    for sev in SEVERITY_ORDER:
        lines.append(f"| {sev.title()} | {counts.get(sev, 0)} |")
    # Any severities outside the known set (unexpected labels) still get counted and shown.
    for sev, count in counts.items():
        if sev not in SEVERITY_ORDER:
            lines.append(f"| {sev.title()} | {count} |")
    lines.append(f"| Total | {total} |")
    lines.append("")

    return "\n".join(lines)


def build_prompt(repo_name: str, enriched_findings: list) -> str:
    code_findings = [f for f in enriched_findings if f.get("finding_type") == "code"]
    dep_findings  = [f for f in enriched_findings if f.get("finding_type") == "dependency"]

    def simplify_code(f):
        return {
            "file": f.get("file", ""), "line": f.get("line", ""),
            "severity": f.get("severity", ""), "issue": f.get("description", ""),
            "code": f.get("code_snippet", "")[:120]
        }

    def simplify_dep(f):
        return {
            "package": f.get("package", ""), "installed_version": f.get("installed_version", ""),
            "fix_version": f.get("fix_version", "no fix available"),
            "severity": f.get("severity", ""), "cvss_score": f.get("cvss_score"),
            "vuln_ids": f.get("osv_ids", []),
            "description": f.get("vulns", [{}])[0].get("description", "")[:200] if f.get("vulns") else ""
        }

    findings_json = json.dumps({
        "code_issues": [simplify_code(f) for f in code_findings[:30]],
        "dependency_vulnerabilities": [simplify_dep(f) for f in dep_findings[:20]]
    }, indent=2)

    severity_counts = _severity_counts(enriched_findings)

    return f"""You are a senior application security engineer writing a vulnerability report.

You have scanned the repository: **{repo_name}**

Here are the findings in JSON format (for your context only -- these will be
rendered as tables separately, so do not restate individual findings):
{findings_json}

Severity counts: {json.dumps(severity_counts)}

Write ONLY the following two sections, in exactly this format (no other
headings, no findings tables, no restating of individual issues):

===EXEC_SUMMARY===
2-3 sentences summarising the overall security posture. Mention total issues
found and the most critical concern. Plain prose, no heading, no emojis or
decorative symbols.
===RECOMMENDATIONS===
3-5 prioritised action items the developer should do this week, as a Markdown
bullet list. No heading, no emojis or decorative symbols.
===END===

Keep the tone direct, professional, and helpful, not alarmist. Focus on
actionable fixes. Do not invent findings that aren't in the JSON.
"""


def parse_llm_sections(raw_text: str) -> dict:
    """Splits the LLM's marker-delimited output into exec_summary and
    recommendations. Falls back gracefully (dumping the whole response into
    exec_summary) if the model didn't follow the marker format exactly, so a
    formatting slip degrades the report rather than crashing it.
    """
    exec_summary = ""
    recommendations = ""

    try:
        after_exec = raw_text.split("===EXEC_SUMMARY===", 1)[1]
        exec_part, rest = after_exec.split("===RECOMMENDATIONS===", 1)
        exec_summary = exec_part.strip()
        recommendations = rest.split("===END===", 1)[0].strip()
    except (IndexError, ValueError):
        exec_summary = raw_text.strip()
        recommendations = ""

    return {"exec_summary": exec_summary, "recommendations": recommendations}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Security Report: {repo_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f1117; color: #e2e4e9; line-height: 1.7; padding: 2rem; }}
    .container {{ max-width: 860px; margin: 0 auto; background: #181b22;
      border-radius: 12px; border: 1px solid #2a2d35; padding: 2.5rem 3rem; }}
    .header {{ border-bottom: 1px solid #2a2d35; padding-bottom: 1.5rem; margin-bottom: 2rem; }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
      font-size: 12px; font-weight: 500; margin-right: 6px; }}
    .badge-meta {{ background: #1e2330; color: #7c8391; border: 1px solid #2a2d35; }}
    h1 {{ font-size: 1.6rem; font-weight: 600; color: #f0f2f5; margin-bottom: 0.5rem; }}
    h2 {{ font-size: 1.15rem; font-weight: 600; color: #c9ccd3; margin: 2rem 0 0.75rem;
      padding-bottom: 0.4rem; border-bottom: 1px solid #2a2d35; }}
    h3 {{ font-size: 1rem; font-weight: 600; color: #a8abb3; margin: 1.25rem 0 0.5rem; }}
    p  {{ margin-bottom: 0.9rem; color: #c2c5cc; }}
    ul, ol {{ padding-left: 1.5rem; margin-bottom: 0.9rem; color: #c2c5cc; }}
    li {{ margin-bottom: 0.35rem; }}
    code {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #1e2330;
      border: 1px solid #2a2d35; padding: 1px 6px; border-radius: 4px;
      font-size: 0.85em; color: #7dd3fc; }}
    table {{ width: 100%; table-layout: fixed; border-collapse: collapse; margin-bottom: 1rem; font-size: 0.9rem; }}
    th {{ background: #1e2330; color: #a8abb3; font-weight: 500; padding: 8px 12px;
      text-align: left; border: 1px solid #2a2d35; overflow-wrap: break-word; word-break: break-word; }}
    td {{ padding: 8px 12px; border: 1px solid #2a2d35; color: #c2c5cc;
      overflow-wrap: break-word; word-break: break-word; }}
    tr:nth-child(even) td {{ background: #1a1d24; }}
    .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #2a2d35;
      font-size: 0.8rem; color: #4a4e5a; }}
    strong {{ color: #e2e4e9; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Security Scan Report</h1>
      <div style="margin-top:0.75rem">
        <span class="badge badge-meta">repo: {repo_name}</span>
        <span class="badge badge-meta">generated: {timestamp}</span>
        <span class="badge badge-meta">powered by Llama 3 - Groq</span>
      </div>
    </div>
    {content}
    <div class="footer">
      Generated by Vulnerability Finder Agent - bandit + pip-audit + OSV.dev - {timestamp}
    </div>
  </div>
</body>
</html>"""


def _call_groq(prompt: str, repo_name: str):
    """Runs the actual Groq LLM call, logged as a Langfuse generation if enabled."""
    client = Groq()  # picks up GROQ_API_KEY from environment automatically
    model_name = "llama-3.3-70b-versatile"

    if not LANGFUSE_ENABLED:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a senior application security engineer. Write clear, accurate, actionable security reports in Markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response

    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="groq-security-report",
        model=model_name,
        input=prompt,
        metadata={"repo_name": repo_name},
    ) as generation:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a senior application security engineer. Write clear, accurate, actionable security reports in Markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        usage = response.usage
        generation.update(
            output=response.choices[0].message.content,
            usage_details={
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
            } if usage else None,
        )
    return response


def build_fix_results_markdown(repo_name: str, code_fixes: list, withheld_fixes: list, dependency_fixes: list) -> str:
    """Builds a Markdown section documenting what happened during fix
    generation -- which findings got a CONFIRMED fix, which were
    WITHHELD and why, and the full critic/verification detail behind
    each decision. This is appended to the security report AFTER fix
    generation completes (the report itself is written before fixes
    are even attempted, so this can't be baked into the original
    prompt/generation step -- see append_fix_results_to_report()).

    This intentionally surfaces the SAME detail that previously only
    lived in proposed_fixes/SUMMARY.md and the PR approval prompt --
    anyone reading only the security report (not the PR flow, not the
    SUMMARY.md file) previously had no visibility into critic findings,
    withheld fixes, or why a fix was rejected. That's a real gap if the
    report is the artifact that gets shared with stakeholders who never
    see the terminal or approve the PR themselves.
    """
    total_findings_with_fix_attempt = len(code_fixes) + len(withheld_fixes)

    if total_findings_with_fix_attempt == 0 and not dependency_fixes:
        return "\n".join([
            "\n---\n", "## Verification Methodology\n",
            "No automated fixes were attempted for this scan, so no verification was run.\n",
            "\n---\n", "## Fix Results\n",
            "No automated fixes were attempted for this scan.\n",
        ])

    lines = ["\n---\n", "## Verification Methodology\n"]
    lines.append(
        "Verification for each fix includes: a syntax check, a `pyflakes` correctness "
        "check, an independent AI code critic (a different model than the one that wrote "
        "the fix, specifically to avoid a model grading its own work), a ground-truth "
        "re-scan with the original static analyzer, and -- for fixes that pass those "
        "checks -- an isolated attempt to actually import/run the fixed file and, if the "
        "repository has its own tests, run them against the fix. A fix is only marked "
        "**confirmed** if it passes every applicable layer.\n"
    )

    lines.append("\n---\n")
    lines.append("## Fix Results\n")
    lines.append(
        f"An AI-assisted fix was attempted for **{total_findings_with_fix_attempt} file(s)** "
        f"with code findings. Each proposed fix was independently verified before being "
        f"considered for a pull request -- **{len(code_fixes)} confirmed**, "
        f"**{len(withheld_fixes)} withheld** pending manual review.\n"
    )

    if code_fixes:
        lines.append("### Confirmed Fixes -- Delivered\n")
        lines.append(
            "These fixes passed all verification layers and were offered for inclusion "
            "in a pull request. The independent critic's comments on each delivered file "
            "are included below for transparency.\n"
        )
        for fix in code_fixes:
            critique = fix.get("critique") or {}
            lines.append(f"**`{fix['relative_path']}`** -- {fix.get('findings_addressed', '?')} finding(s) addressed")
            lines.append("")  # blank line required so markdown renders the following as a list, not a run-on paragraph
            if critique.get("score") is not None:
                lines.append(f"- Critic score: {critique.get('score')}/10 ({critique.get('verdict')})")
            verification = fix.get("verification") or {}
            _append_verification_lines(lines, verification)
            non_critical = [p for p in critique.get("problems", []) if isinstance(p, dict) and p.get("severity") != "critical"]
            if non_critical:
                lines.append("- Non-blocking notes from the critic:")
                for p in non_critical:
                    lines.append(f"  - {p.get('text', '')}")
            if critique.get("suggestions"):
                lines.append(f"- Critic's suggestions: {critique.get('suggestions')}")
            lines.append("")

    if withheld_fixes:
        lines.append("### Withheld Fixes -- Require Manual Review\n")
        lines.append(
            "These fixes did **not** pass verification -- either the original vulnerability "
            "was still detectable after the fix, or the independent critic identified a "
            "critical problem. They were **excluded from the pull request by default** and "
            "require manual review; a human reviewer would need to explicitly override this "
            "to include them.\n"
        )
        for fix in withheld_fixes:
            critique = fix.get("critique") or {}
            lines.append(f"**`{fix['relative_path']}`** -- {fix.get('findings_addressed', '?')} finding(s) addressed, verdict: **{critique.get('verdict', 'unknown')}**")
            lines.append("")  # blank line required so markdown renders the following as a list, not a run-on paragraph
            if critique.get("score") is not None:
                lines.append(f"- Critic score: {critique.get('score')}/10")
            verification = fix.get("verification") or {}
            _append_verification_lines(lines, verification)
            critical_problems = [p for p in critique.get("problems", []) if isinstance(p, dict) and p.get("severity") == "critical"]
            if critical_problems:
                lines.append("- Critical problems identified:")
                for p in critical_problems:
                    lines.append(f"  - {p.get('text', '')}")
            moderate_problems = [p for p in critique.get("problems", []) if isinstance(p, dict) and p.get("severity") == "moderate"]
            if moderate_problems:
                lines.append("- Other problems noted:")
                for p in moderate_problems:
                    lines.append(f"  - {p.get('text', '')}")
            if critique.get("suggestions"):
                lines.append(f"- Critic's suggestions: {critique.get('suggestions')}")
            lines.append(f"- Proposed (unmerged) file available at: `{fix.get('fixed_path', 'n/a')}` for manual review")
            lines.append("")

    if dependency_fixes:
        lines.append("### Dependency Fixes\n")
        lines.append(
            "Dependency version bumps are deterministic (the fix version comes directly "
            "from OSV.dev, not an AI model) and are not subject to the critic/verification "
            "process above.\n"
        )
        for fix in dependency_fixes:
            tag = "direct" if fix.get("fix_type") == "direct" else "transitive (new pin)"
            lines.append(f"- [{tag}] `{fix['package']}`: `{fix['old_line']}` → `{fix['new_line']}`")
        lines.append("")

    return "\n".join(lines)


def _append_verification_lines(lines: list, verification: dict):
    """Helper: appends a bullet per verification layer's result, if that
    layer actually ran (skipped layers show a distinct, honest label
    rather than being silently omitted or conflated with a real pass)."""
    if not verification:
        return

    if verification.get("bandit_still_flags") is True:
        lines.append(f"- Fail -- Bandit re-scan: original issue rule(s) `{verification.get('remaining_issues')}` still detected")
    elif verification.get("syntax_valid"):
        lines.append("- Pass -- Bandit re-scan: original issue no longer detected")

    pyflakes = verification.get("pyflakes")
    if pyflakes and pyflakes.get("has_critical"):
        lines.append("- Fail -- pyflakes: found a real correctness bug (undefined name or similar)")
    elif pyflakes is not None:
        lines.append("- Pass -- pyflakes: no correctness-breaking issues")

    rule_coverage = verification.get("rule_coverage")
    if rule_coverage:
        if not rule_coverage.get("applicable", True):
            # Nothing to check (no rule IDs, or critic unavailable) --
            # distinct from "Covered", since we didn't actually verify
            # anything here.
            lines.append("- N/A -- Rule coverage: no rule IDs to cross-check, or critic unavailable")
        elif rule_coverage.get("ok"):
            lines.append("- Covered -- Rule coverage: critic explicitly confirmed every original finding rule was addressed")
        else:
            lines.append(f"- Partial -- Rule coverage: critic did not confirm rule(s) `{rule_coverage.get('missing')}` were resolved -- verdict downgraded, treat as unconfirmed")

    import_check = verification.get("import_check")
    if import_check:
        if import_check.get("skipped"):
            lines.append(f"- Skipped -- Isolated import check: skipped ({import_check.get('error') or 'not applicable'})")
        elif import_check.get("import_ok"):
            lines.append("- Pass -- Isolated import check: fixed file imports/runs without error")
        else:
            lines.append(f"- Fail -- Isolated import check: fixed file does not import/run -- {import_check.get('error')}")

    existing_tests = verification.get("existing_tests")
    if existing_tests:
        tests_found = existing_tests.get("tests_found")
        tests_passed = existing_tests.get("tests_passed")
        summary = str(existing_tests.get("summary", ""))

        if summary.startswith("skipped") or not tests_found:
            # No tests exist in the repo at all, or we couldn't even set
            # up the sandbox to check -- a real absence, not a result.
            lines.append(f"- Skipped -- Existing test suite: {summary or 'none found in repository'}")
        elif tests_passed is True:
            lines.append("- Pass -- Existing test suite: passes with this fix applied")
        elif tests_passed is False:
            lines.append(f"- Fail -- Existing test suite: fails with this fix applied -- {summary}")
        else:
            # tests_found=True but tests_passed=None: baseline and fixed
            # both failed for the same environment reason (or the run
            # timed out) -- a real result, just not attributable to the
            # fix either way. Distinct from "Skipped" (no tests exist).
            lines.append(f"- Inconclusive -- Existing test suite: {summary}")


def append_fix_results_to_report(report_path: str, report_markdown: str, repo_name: str,
                                   code_fixes: list, withheld_fixes: list, dependency_fixes: list) -> dict:
    """Appends the fix-results section to an already-generated report,
    re-renders to HTML, and overwrites the same file on disk. Called
    AFTER fix_generate_node runs, since fix data doesn't exist yet at
    the point the original report is generated (report generation is
    the first human-approval checkpoint, before any fixes are
    attempted).

    Returns {"success": bool, "report_markdown": str (updated),
    "report_path": str, "error": optional str}. Failure here is
    non-fatal to the overall pipeline -- the original report on disk is
    left untouched if anything goes wrong, so a bug in this step never
    destroys the already-generated, already-approved report.
    """
    if not report_path or not os.path.exists(report_path):
        return {"success": False, "report_markdown": report_markdown, "report_path": report_path,
                 "error": f"Report path not found, cannot append fix results: {report_path}"}

    try:
        fix_section_md = build_fix_results_markdown(repo_name, code_fixes, withheld_fixes, dependency_fixes)
        base_markdown = report_markdown or ""

        if FIX_RESULTS_MARKER in base_markdown:
            # Normal path: splice into the placeholder left between
            # "Findings Breakdown" and "Recommendations" so the final
            # heading order stays Header -> Exec Summary -> Findings
            # Breakdown -> Verification Methodology -> Fix Results ->
            # Recommendations -> Scan Summary.
            updated_markdown = base_markdown.replace(FIX_RESULTS_MARKER, fix_section_md)
        else:
            # Defensive fallback for older reports generated before this
            # marker existed -- append at the end rather than failing.
            updated_markdown = base_markdown + "\n" + fix_section_md

        content_html = md_lib.markdown(updated_markdown, extensions=["tables", "fenced_code"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_html = HTML_TEMPLATE.format(repo_name=repo_name, timestamp=timestamp, content=content_html)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        return {"success": True, "report_markdown": updated_markdown, "report_path": report_path, "error": None}

    except Exception as e:
        return {"success": False, "report_markdown": report_markdown, "report_path": report_path,
                 "error": f"Failed to append fix results to report: {e}"}


@observe(name="generate_report")
def generate_report(repo_name: str, enriched_findings: list, output_dir: str = None) -> dict:
    prompt = build_prompt(repo_name, enriched_findings)

    if LANGFUSE_ENABLED:
        get_client().update_current_trace(
            metadata={
                "repo_name": repo_name,
                "finding_count": str(len(enriched_findings)),
            }
        ) if hasattr(get_client(), "update_current_trace") else None

    try:
        response = _call_groq(prompt, repo_name)
        raw_llm_output = response.choices[0].message.content

    except Exception as e:
        return {"success": False, "report_markdown": None, "report_path": None,
                 "error": f"Groq API error: {e}"}

    sections = parse_llm_sections(raw_llm_output)
    findings_breakdown_md = build_findings_breakdown_markdown(enriched_findings, repo_name)
    scan_summary_md = build_scan_summary_markdown(enriched_findings)

    report_markdown = "\n\n".join([
        f"# Security Scan Report: {repo_name}",
        "## Executive Summary\n\n" + (sections["exec_summary"] or "No summary available."),
        findings_breakdown_md,
        FIX_RESULTS_MARKER,
        "## Recommendations\n\n" + (sections["recommendations"] or "No recommendations available."),
        scan_summary_md,
    ])

    content_html = md_lib.markdown(report_markdown, extensions=["tables", "fenced_code"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_ts    = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_html = HTML_TEMPLATE.format(repo_name=repo_name, timestamp=timestamp, content=content_html)

    save_dir = output_dir if output_dir else REPORTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    safe_name   = repo_name.replace(" ", "_").replace("/", "_")
    output_path = os.path.join(save_dir, f"report_{safe_name}_{file_ts}.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return {"success": True, "report_markdown": report_markdown, "report_path": output_path, "error": None}