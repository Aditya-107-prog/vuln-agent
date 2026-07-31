"""
Tool 3: Code Scanner (bandit for Python, semgrep for Go/Java)
---------------------------------------------------------------
Runs a language-appropriate static analyzer on a directory (or single
file) and returns a clean, structured list of findings in ONE common
schema, regardless of which underlying tool produced them -- callers
elsewhere in the pipeline (cve_enrich, fix_generator, report_generator)
already only depend on this schema, not on which scanner ran.

Python: bandit, checks for things like:
- Hardcoded passwords / secrets
- SQL injection risks
- Use of dangerous functions (eval, exec, pickle)
- Weak cryptography (MD5, SHA1)
- Insecure HTTP usage
- And ~100 other patterns

Go / Java: semgrep, using LOCAL rule files bundled in
tools/semgrep_rules/ (go.yaml, java.yaml) rather than semgrep's
registry (p/golang, p/java, etc). This is deliberate: registry rulesets
require network access to semgrep.dev and, for some rulesets/features,
a semgrep login -- neither should be a hard dependency for this
pipeline to run. Local rules also mean the exact ruleset in use is
version-controlled with the rest of this project, not silently
changing whenever the registry updates. The tradeoff is coverage: this
starter ruleset is intentionally small (SQL injection, shell/command
injection, weak hashing) rather than semgrep's full registry breadth --
expand tools/semgrep_rules/*.yaml as more patterns are needed.
"""

import subprocess
import json
import os
import glob

from logging_config import get_logger

logger = get_logger(__name__)

SEMGREP_RULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semgrep_rules")
SEMGREP_RULE_FILES = {
    "go": os.path.join(SEMGREP_RULES_DIR, "go.yaml"),
    "java": os.path.join(SEMGREP_RULES_DIR, "java.yaml"),
}
SEMGREP_TIMEOUT_SECONDS = 120

# Rule IDs that are structurally unfixable -- no code change silences
# them, so treating them as "needs a fix" only produces fixes that
# bandit will always still flag, forcing a permanent "fail" verdict
# regardless of how good the fix actually is:
#   B404 -- fires the moment a file contains `import subprocess` at
#           all, regardless of how the module is used afterward. Even
#           the textbook-safe subprocess pattern (shell=False, list of
#           args, no string concat) still trips this.
#   B101 -- fires on any `assert` statement. This is normal, correct,
#           idiomatic pytest usage in test files -- bandit's actual
#           concern (asserts silently stripped by python -O, so never
#           rely on them for production-code security checks) doesn't
#           apply to test assertions.
# Tradeoff worth knowing: skipping B101 globally also silences it for
# `assert` misused in non-test production code, which IS a legitimate
# thing to flag. If this pipeline ever scans repos with meaningful
# production code outside test files, consider excluding B101 only for
# test file paths instead (bandit's --skip is global, not path-scoped,
# so that would need post-parse filtering by filename pattern).
BANDIT_SKIP_TESTS = "B404,B101"

# B603 ("subprocess call - check for execution of untrusted input") is
# NOT always unfixable the way B404/B101 are -- bandit itself marks it
# LOW confidence specifically because it can't tell whether the input
# is trusted, and a HIGH/MEDIUM confidence B603 can still indicate a
# real problem. So instead of skipping the rule outright, only LOW
# confidence B603 findings are dropped after the scan, below.


def _run_python_scan(repo_path: str, recursive: bool = True) -> dict:
    """
    Run bandit on a directory (recursive=True) or a single file
    (recursive=False) and return structured findings.

    recursive=False is used by fix_generator.py to re-scan a single
    proposed-fix file in isolation, as a ground-truth check that a
    fix actually removed the bandit rule it was meant to fix -- not
    just something that looks fixed to an LLM critic.

    Args:
        repo_path: Path to the directory (or file) to scan
        recursive: whether to pass bandit's -r flag

    Returns:
        {
            "success": True,
            "findings": [
                {
                    "file": "src/app.py",
                    "line": 42,
                    "severity": "HIGH",
                    "confidence": "MEDIUM",
                    "issue": "B105",
                    "description": "Possible hardcoded password: 'secret123'",
                    "code_snippet": "password = 'secret123'"
                },
                ...
            ],
            "total": 5,
            "error": None
        }
    """

    logger.info(f"Running bandit on {repo_path} (recursive={recursive})")

    try:
        cmd = ["bandit"]
        if recursive:
            cmd.append("-r")
        cmd.append(repo_path)
        cmd += [
            "-f", "json",   # output as JSON so we can parse it
            "-q",           # quiet — suppress progress output
            "--exit-zero",  # don't fail with non-zero exit code if findings exist
                            # (we handle findings ourselves)
            "--skip", BANDIT_SKIP_TESTS,
            # ^ applies to BOTH the full recursive scan and the
            # single-file ground-truth re-scan fix_generator.py does
            # per fix attempt -- important, since if these rules were
            # only skipped on the initial scan but still checked during
            # re-verification, a fix could still get force-failed by a
            # rule the report never even showed as needing a fix.
        ]
        if recursive:
            cmd += ["--exclude", ".venv,venv,env,.env,node_modules,dist,build"]
            # ^ skip virtual envs and build folders — not relevant for a
            # single temp file, only meaningful on a real directory scan

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            # Explicit UTF-8 instead of text=True's platform-default
            # encoding, which is cp1252 on Windows, not UTF-8 -- a
            # non-cp1252 byte anywhere in bandit's output (e.g. a
            # Unicode character in a docstring/comment bandit quotes
            # back in code_snippet) crashes the subprocess reader
            # thread silently and leaves result.stdout as None instead
            # of raising. See dep_scanner.py's run_go_dependency_scan
            # for the real incident that surfaced this on Windows.
            encoding="utf-8",
            errors="replace",
        )

        # bandit outputs JSON to stdout
        output = result.stdout or ""
        if not output.strip():
            return {
                "success": True,
                "findings": [],
                "total": 0,
                "error": None
            }

        raw = json.loads(output)

        # --- Parse bandit's output into our clean format ---
        findings = []
        for issue in raw.get("results", []):
            findings.append({
                "file": issue.get("filename", "unknown"),
                "line": issue.get("line_number", 0),
                "severity": issue.get("issue_severity", "UNKNOWN").upper(),
                # CONFIDENCE = how sure bandit is this is actually a problem
                # HIGH confidence = very likely a real issue
                # LOW confidence = might be a false positive
                "confidence": issue.get("issue_confidence", "UNKNOWN").upper(),
                "issue": issue.get("test_id", "unknown"),  # e.g. B105, B201
                "description": issue.get("issue_text", ""),
                "code_snippet": issue.get("code", "").strip()
            })

        # Sort by severity: HIGH first, then MEDIUM, then LOW
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
        findings.sort(key=lambda x: severity_order.get(x["severity"], 3))

        # Drop LOW-confidence B603 findings -- see BANDIT_SKIP_TESTS
        # comment above for why this one can't just be blanket-skipped
        # via --skip like B404/B101 (a HIGH/MEDIUM confidence B603 can
        # still be a real problem, so only the low-confidence noise is
        # filtered here).
        before_count = len(findings)
        findings = [f for f in findings if not (f["issue"] == "B603" and f["confidence"] == "LOW")]
        dropped = before_count - len(findings)
        if dropped:
            logger.info(f"Filtered out {dropped} low-confidence B603 finding(s)")

        logger.info(f"Found {len(findings)} code issues")

        return {
            "success": True,
            "findings": findings,
            "total": len(findings),
            "error": None
        }

    except FileNotFoundError:
        logger.error("bandit is not installed")
        return {
            "success": False,
            "findings": [],
            "total": 0,
            "error": "bandit is not installed. Run: pip install bandit"
        }

    except subprocess.TimeoutExpired:
        logger.error(f"bandit timed out after 120 seconds on {repo_path}")
        return {
            "success": False,
            "findings": [],
            "total": 0,
            "error": "bandit timed out after 120 seconds"
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse bandit JSON output: {e}")
        return {
            "success": False,
            "findings": [],
            "total": 0,
            "error": f"Failed to parse bandit JSON output: {e}"
        }


# ---------------------------------------------------------------------------
# Go / Java: semgrep wrapper (local rule files, see module docstring)
# ---------------------------------------------------------------------------
def _semgrep_severity_map(sev: str) -> str:
    # semgrep's severities (INFO/WARNING/ERROR) don't share bandit's
    # vocabulary (LOW/MEDIUM/HIGH) -- normalize to bandit's so
    # report_generator's severity counts/sorting work unmodified across
    # both scanner families.
    return {"ERROR": "HIGH", "WARNING": "MEDIUM", "INFO": "LOW"}.get((sev or "").upper(), "UNKNOWN")


def _extract_code_snippet(file_path: str, line_number: int) -> str:
    """semgrep's extra.lines field returns the literal string
    "requires login" when run without a semgrep.dev account (true for
    the local-rules-only setup this pipeline uses) -- so the code
    snippet has to be read directly from the scanned file instead of
    trusting that field. Best-effort: returns "" if the file can't be
    read (e.g. it's already been cleaned up, as happens with the temp
    files used for single-file re-scans during fix verification)."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1].strip()
    except Exception:
        pass
    return ""


def _run_semgrep_scan(repo_path: str, language: str, recursive: bool = True) -> dict:
    """Runs semgrep with this project's LOCAL rule file for `language`
    (see SEMGREP_RULE_FILES / module docstring for why local, not
    registry, rules are used). Same recursive=True/False split as
    _run_python_scan: True for a full repo scan, False for a
    single-file ground-truth re-scan during fix verification.

    Returns the SAME schema _run_python_scan returns, plus a
    "language" key on each finding (bandit findings don't get this key
    added, to avoid changing behavior for any existing Python-only
    caller that doesn't expect it -- callers that need to distinguish
    should use .get("language", "python") rather than assuming its
    presence).
    """
    rule_file = SEMGREP_RULE_FILES.get(language)
    if not rule_file or not os.path.exists(rule_file):
        logger.error(f"No semgrep rule file configured for language={language!r} (looked for {rule_file})")
        return {"success": False, "findings": [], "total": 0, "error": f"no semgrep rules configured for language '{language}'"}

    logger.info(f"Running semgrep ({language}) on {repo_path} (recursive={recursive})")

    try:
        cmd = [
            "semgrep",
            "--config", rule_file,
            "--json",
            "--quiet",
            "--metrics=off",
        ]
        cmd.append(repo_path)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SEMGREP_TIMEOUT_SECONDS,
            # Explicit UTF-8 -- this is the exact call that crashed in
            # the real Windows incident that prompted this fix (see
            # dep_scanner.py's run_go_dependency_scan for the full
            # explanation). text=True alone decodes using the
            # platform-default locale encoding, which is cp1252 on
            # Windows rather than UTF-8; semgrep's JSON output
            # (message/code_snippet text pulled from arbitrary scanned
            # source files) can easily contain a byte that isn't valid
            # cp1252, which silently kills the subprocess reader
            # thread and leaves result.stdout as None instead of
            # raising -- not just a cosmetic decode warning.
            encoding="utf-8",
            errors="replace",
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # A genuine CLI failure (bad flags, crash) exits non-zero with
        # empty/non-JSON stdout -- must NOT be treated the same as "ran
        # fine, found nothing", or a broken invocation silently looks
        # identical to a clean scan. Exit code 1 is semgrep's normal
        # "ran fine, findings exist" code with --json though, so only
        # treat 2+ (or empty stdout with any non-zero code) as fatal.
        if not stdout.strip():
            if result.returncode not in (0, 1):
                logger.error(f"semgrep ({language}) CLI failed (exit {result.returncode}): {stderr.strip()[:500]}")
                return {"success": False, "findings": [], "total": 0, "error": f"semgrep CLI failed (exit {result.returncode}): {stderr.strip()[:300]}"}
            return {"success": True, "findings": [], "total": 0, "error": None}

        raw = json.loads(stdout)

        errors = raw.get("errors") or []
        fatal_errors = [e for e in errors if e.get("level") == "error" and "requires login" not in str(e.get("message", ""))]
        if fatal_errors and not raw.get("results"):
            # A real failure (bad rule file, parser crash, etc), not
            # just the benign "requires login" notice on the lines
            # field -- surface it rather than silently returning zero
            # findings, which would look identical to "scanned clean".
            msg = "; ".join(e.get("message", "") for e in fatal_errors)
            logger.error(f"semgrep ({language}) reported error(s): {msg}")
            return {"success": False, "findings": [], "total": 0, "error": msg}

        findings = []
        for r in raw.get("results", []):
            file_path = r.get("path", "unknown")
            line = r.get("start", {}).get("line", 0)
            metadata = r.get("extra", {}).get("metadata", {}) or {}
            findings.append({
                "file": file_path,
                "line": line,
                "severity": _semgrep_severity_map(r.get("extra", {}).get("severity")),
                # semgrep doesn't have bandit's separate confidence
                # concept -- everything a rule fires is treated as
                # MEDIUM confidence by default (no equivalent
                # HIGH/LOW distinction to draw from without a second
                # signal), consistent with how the rest of the
                # pipeline expects a "confidence" key to exist even if
                # this scanner family can't populate it as precisely
                # as bandit does.
                "confidence": "MEDIUM",
                # rule_id (from our own metadata, e.g. "GO-SQLI-001")
                # is what fix_generator/critic treat as the stable
                # identifier -- semgrep's own check_id is a dotted
                # rule-file path (tools.semgrep_rules.go-sql-...) which
                # is an implementation detail, not a stable public ID.
                "issue": metadata.get("rule_id", r.get("check_id", "unknown")),
                "description": r.get("extra", {}).get("message", ""),
                "code_snippet": _extract_code_snippet(file_path, line),
                "language": language,
            })

        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
        findings.sort(key=lambda x: severity_order.get(x["severity"], 3))

        logger.info(f"Found {len(findings)} {language} issue(s) via semgrep")
        return {"success": True, "findings": findings, "total": len(findings), "error": None}

    except FileNotFoundError:
        logger.error("semgrep is not installed")
        return {"success": False, "findings": [], "total": 0, "error": "semgrep is not installed. Run: pip install semgrep"}

    except subprocess.TimeoutExpired:
        logger.error(f"semgrep timed out after {SEMGREP_TIMEOUT_SECONDS} seconds on {repo_path}")
        return {"success": False, "findings": [], "total": 0, "error": f"semgrep timed out after {SEMGREP_TIMEOUT_SECONDS} seconds"}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse semgrep JSON output: {e}")
        return {"success": False, "findings": [], "total": 0, "error": f"Failed to parse semgrep JSON output: {e}"}


# ---------------------------------------------------------------------------
# Unified dispatcher -- this is the function the rest of the pipeline calls
# ---------------------------------------------------------------------------
_EXTENSION_TO_LANGUAGE = {".py": "python", ".go": "go", ".java": "java"}


def _infer_single_file_language(file_path: str) -> str:
    """Used for recursive=False single-file re-scans during fix
    verification, where repo_path is actually one file, not a
    directory -- language can't come from lang_detect's marker-file
    walk (no go.mod next to a temp file), so infer from the extension
    instead. Defaults to "python" if unrecognized, preserving the
    pre-multi-language behavior for any caller that doesn't pass an
    explicit language.
    """
    _, ext = os.path.splitext(file_path)
    return _EXTENSION_TO_LANGUAGE.get(ext.lower(), "python")


def run_code_scan(repo_path: str, recursive: bool = True, language: str = None) -> dict:
    """Unified entrypoint -- dispatches to bandit (Python) or semgrep
    (Go/Java) based on `language`, or auto-detects when not given.

    Backward compatible with every existing call site: callers that
    never passed `language` (the entire pipeline, before this session)
    keep working unchanged --
      - recursive=True, no language: auto-detects ALL languages present
        via lang_detect and runs each scanner, merging results into one
        findings list (each finding tagged "language" so downstream
        code can tell them apart). A pure-Python repo behaves exactly
        as before: one language detected, one scanner run, same
        findings shape bandit already returned (Python findings do NOT
        get a "language" key added, so any existing code doing an exact
        dict-shape comparison in tests is unaffected).
      - recursive=False, no language: this is always a single-file
        ground-truth re-scan during fix verification (see
        fix_generator.py's _rescan_with_bandit) -- language is inferred
        from the file extension.

    Returns: {"success": bool, "findings": [...], "total": int, "error": str|None}
    """
    if language:
        if language == "python":
            return _run_python_scan(repo_path, recursive=recursive)
        return _run_semgrep_scan(repo_path, language, recursive=recursive)

    if not recursive:
        inferred = _infer_single_file_language(repo_path)
        if inferred == "python":
            return _run_python_scan(repo_path, recursive=False)
        return _run_semgrep_scan(repo_path, inferred, recursive=False)

    # Full repo scan, no language specified -- detect and run all.
    from tools.lang_detect import detect_languages
    languages = detect_languages(repo_path)
    if not languages:
        logger.info(f"No recognized language detected in {repo_path} -- nothing to scan")
        return {"success": True, "findings": [], "total": 0, "error": None}

    all_findings = []
    errors = []
    for lang in languages:
        if lang == "python":
            result = _run_python_scan(repo_path, recursive=True)
        elif lang in SEMGREP_RULE_FILES:
            result = _run_semgrep_scan(repo_path, lang, recursive=True)
        else:
            logger.info(f"[code_scanner] Detected language '{lang}' has no scanner configured yet -- skipping")
            continue

        if not result["success"]:
            errors.append(f"{lang}: {result.get('error')}")
        all_findings.extend(result["findings"])

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
    all_findings.sort(key=lambda x: severity_order.get(x.get("severity"), 3))

    return {
        "success": len(errors) == 0,
        "findings": all_findings,
        "total": len(all_findings),
        "error": "; ".join(errors) if errors else None,
    }


# --- Quick sanity check ---
if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"=== Scanning: {target} ===\n")
    result = run_code_scan(target)

    if result["success"]:
        print(f"Total findings: {result['total']}\n")
        for f in result["findings"][:5]:  # show first 5
            print(f"[{f['severity']}] {f['file']}:{f['line']}")
            print(f"  Issue: {f['description']}")
            print(f"  Code:  {f['code_snippet'][:80]}")
            print()
    else:
        print(f"Error: {result['error']}")