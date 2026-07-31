"""
Tool: Dependency Scanner (pip-audit for Python, govulncheck for Go,
osv-scanner for Java)
--------------------------------------------------------------------
Python path (pip-audit) is UNCHANGED from before -- see the original
FIXES notes below, still accurate. Go and Java are new, separate
functions (run_go_dependency_scan, run_java_dependency_scan) rather
than folded into run_dependency_scan, so the existing Python call path
and its behavior are not touched by this addition.

FIXES applied (Python / pip-audit path, unchanged):
1. Searches common subfolders (app/, src/, backend/, api/, server/, service/,
   web/) one level deep, not just the repo root -- many real repos nest
   their requirements.txt (e.g. we45/Vulnerable-Flask-App keeps it in app/).
2. CRITICAL FIX: the previous version called `pip-audit --path repo_path`,
   but --path means "audit an installed environment at this path" (like a
   venv), NOT "read this requirements file's contents." It never actually
   told pip-audit to parse requirements.txt. Now uses `-r <requirements_file>`
   which is the correct flag for auditing a requirements file's declared
   dependencies. This likely explains false "0 vulnerable packages" results
   on repos that do have real known-vulnerable pinned dependencies.

Go (govulncheck): needs the Go toolchain installed. Unlike semgrep in
code_scanner.py, there's no way around this -- govulncheck's actual
value proposition IS build-aware reachability analysis (does your code
actually CALL the vulnerable function, not just import the package),
which requires real compilation-adjacent analysis, not a lockfile
lookup. If govulncheck isn't installed, run_go_dependency_scan returns
a clean "not installed" error rather than silently reporting 0 findings.

Java (osv-scanner): chosen over OWASP dependency-check specifically
because it queries OSV.dev -- the SAME database cve_enricher.py already
uses for Python findings -- rather than NVD, which would give Java
findings inconsistent IDs/severity conventions from everything else in
the report. osv-scanner also doesn't need a Maven/Gradle build to run,
consistent with this pipeline's general avoid-a-build-step preference.
"""

import subprocess
import json
import os

from logging_config import get_logger

logger = get_logger(__name__)

GOVULNCHECK_TIMEOUT_SECONDS = 180
OSV_SCANNER_TIMEOUT_SECONDS = 180

REQ_FILENAMES = ["requirements.txt", "requirements.in"]
COMMON_SUBDIRS = ["", "app", "src", "backend", "api", "server", "service", "web"]




def _find_requirements_file(repo_path: str) -> str:
    """Searches repo root and one level of common subfolders for a
    requirements file. Returns the full path if found, else None."""
    for subdir in COMMON_SUBDIRS:
        for filename in REQ_FILENAMES:
            candidate = os.path.join(repo_path, subdir, filename) if subdir else os.path.join(repo_path, filename)
            if os.path.exists(candidate):
                return candidate
    return None


def run_dependency_scan(repo_path: str) -> dict:
    req_file_path = _find_requirements_file(repo_path)

    if not req_file_path:
        return {"success": True, "findings": [], "total": 0,
                 "requirements_found": False, "requirements_file": None, "error": None}

    try:
        result = subprocess.run(
            ["pip-audit", "-r", req_file_path, "--format", "json", "--progress-spinner", "off"],
            capture_output=True, text=True, timeout=240,
            # Explicit UTF-8 instead of relying on text=True's locale
            # default -- on Windows that default is cp1252, not UTF-8,
            # and any non-cp1252 byte in pip-audit's output (accented
            # names, Unicode punctuation in a CVE description, etc.)
            # crashes the subprocess reader thread and leaves
            # result.stdout as None instead of raising cleanly. See
            # run_go_dependency_scan's identical fix for the incident
            # that surfaced this.
            encoding="utf-8", errors="replace",
        )

        output = (result.stdout or "").strip()
        if not output:
            return {"success": True, "findings": [], "total": 0,
                     "requirements_found": True, "requirements_file": req_file_path, "error": None}

        raw = json.loads(output)

        findings = []
        for dep in raw.get("dependencies", []):
            vulns = dep.get("vulns", [])
            if not vulns:
                continue

            fix_version = None
            parsed_vulns = []
            for v in vulns:
                parsed_vulns.append({
                    "id": v.get("id", "unknown"),
                    "description": v.get("description", "No description available")
                })
                fixes = v.get("fix_versions", [])
                if fixes and fix_version is None:
                    fix_version = fixes[0]

            findings.append({
                "package": dep.get("name", "unknown"),
                "installed_version": dep.get("version", "unknown"),
                "fix_version": fix_version,
                "vulns": parsed_vulns
            })

        return {"success": True, "findings": findings, "total": len(findings),
                 "requirements_found": True, "requirements_file": req_file_path, "error": None}

    except FileNotFoundError:
        return {"success": False, "findings": [], "total": 0,
                 "requirements_found": True, "requirements_file": req_file_path,
                 "error": "pip-audit is not installed. Run: pip install pip-audit"}
    except subprocess.TimeoutExpired:
        return {"success": False, "findings": [], "total": 0,
                 "requirements_found": True, "requirements_file": req_file_path,
                 "error": "pip-audit timed out after 240 seconds"}
    except json.JSONDecodeError as e:
        return {"success": False, "findings": [], "total": 0,
                 "requirements_found": True, "requirements_file": req_file_path,
                 "error": f"Failed to parse pip-audit output: {e}"}

# ---------------------------------------------------------------------------
# Go: govulncheck wrapper
# ---------------------------------------------------------------------------
def _find_go_mod(repo_path: str) -> str:
    """Returns the directory containing go.mod, or None. Reuses the
    same common-subdirs search as _find_requirements_file for
    consistency, though govulncheck itself needs a DIRECTORY (module
    root), not a file path, unlike pip-audit's -r flag."""
    for subdir in COMMON_SUBDIRS:
        candidate = os.path.join(repo_path, subdir, "go.mod") if subdir else os.path.join(repo_path, "go.mod")
        if os.path.exists(candidate):
            return os.path.dirname(candidate) if subdir else repo_path
    return None


def _parse_govulncheck_stream(raw_output: str):
    """govulncheck's JSON output is a STREAM of concatenated,
    individually-pretty-printed JSON objects (config/progress/osv/
    finding message envelopes) -- NOT a single JSON array and NOT
    newline-delimited JSON (each object spans many lines). A plain
    json.loads() on the whole thing fails. This walks the string with
    a raw_decode loop, consuming one top-level JSON value at a time.

    Returns (osv_entries: {id: full osv dict}, raw_findings: [finding dicts]).
    """
    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw_output)
    osv_entries = {}
    raw_findings = []
    while idx < n:
        while idx < n and raw_output[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(raw_output, idx)  # let JSONDecodeError propagate to caller
        idx = end
        if "osv" in obj:
            entry = obj["osv"]
            osv_entries[entry.get("id")] = entry
        elif "finding" in obj:
            raw_findings.append(obj["finding"])
        # "config"/"progress"/"SBOM" messages are ignored -- not
        # findings-relevant.
    return osv_entries, raw_findings


def run_go_dependency_scan(repo_path: str) -> dict:
    """Runs govulncheck against a Go module (needs `go.mod` and the Go
    toolchain -- see module docstring for why this can't avoid needing
    a build the way semgrep/osv-scanner do).

    Returns the SAME shape run_dependency_scan (pip-audit) already
    returns -- {"package","installed_version","fix_version","vulns":[...]}}
    per finding -- so it flows through cve_enrich/report_generator
    unchanged. Adds "ecosystem": "Go" (for cve_enricher.py's OSV
    query) plus, when govulncheck's call-graph analysis actually
    traced a real call path, "reachable": True and
    "call_trace_summary": a human-readable call chain. Findings where
    only the vulnerable package is imported (not actually called) get
    "reachable": False -- this is govulncheck's core value versus a
    plain lockfile scan, and is preserved here rather than discarded,
    even though severity/CVSS still comes from a normal OSV.dev query
    like every other language (govulncheck's own OSV entries don't
    reliably carry a CVSS score, so re-querying OSV.dev keeps severity
    data consistent across Python/Go/Java findings).
    """
    go_mod_dir = _find_go_mod(repo_path)
    if not go_mod_dir:
        return {"success": True, "findings": [], "total": 0,
                "requirements_found": False, "requirements_file": None, "error": None}

    go_mod_path = os.path.join(go_mod_dir, "go.mod")

    try:
        result = subprocess.run(
            ["govulncheck", "-C", go_mod_dir, "-format", "json", "./..."],
            capture_output=True, text=True, timeout=GOVULNCHECK_TIMEOUT_SECONDS,
            # Explicit UTF-8 -- see pip-audit's call above for why
            # text=True's platform-default encoding (cp1252 on
            # Windows) is unsafe here. This is the exact call that
            # crashed in the real incident that prompted this fix: a
            # non-cp1252 byte in govulncheck's JSON output (a CVE
            # description containing a Unicode character) killed the
            # subprocess reader thread, leaving result.stdout as None
            # rather than raising -- .strip() on None then crashed the
            # whole node.
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return {"success": False, "findings": [], "total": 0,
                "requirements_found": True, "requirements_file": go_mod_path,
                "error": "govulncheck is not installed. Run: go install golang.org/x/vuln/cmd/govulncheck@latest"}
    except subprocess.TimeoutExpired:
        return {"success": False, "findings": [], "total": 0,
                "requirements_found": True, "requirements_file": go_mod_path,
                "error": f"govulncheck timed out after {GOVULNCHECK_TIMEOUT_SECONDS} seconds"}

    output = (result.stdout or "").strip()
    if not output:
        # govulncheck exits 3 when vulnerabilities ARE found (its own
        # documented CI convention) -- so empty stdout at any code
        # OTHER than 0 (clean) or 3 (vulns found, but somehow no
        # stdout -- shouldn't normally happen) means a real failure,
        # not "nothing to report".
        if result.returncode not in (0, 3):
            return {"success": False, "findings": [], "total": 0,
                    "requirements_found": True, "requirements_file": go_mod_path,
                    "error": f"govulncheck failed (exit {result.returncode}): {result.stderr.strip()[:500]}"}
        return {"success": True, "findings": [], "total": 0,
                "requirements_found": True, "requirements_file": go_mod_path, "error": None}

    try:
        osv_entries, raw_findings = _parse_govulncheck_stream(output)
    except json.JSONDecodeError as e:
        return {"success": False, "findings": [], "total": 0,
                "requirements_found": True, "requirements_file": go_mod_path,
                "error": f"Failed to parse govulncheck JSON stream: {e}"}

    # Dedupe: govulncheck emits multiple "finding" messages for the
    # SAME (osv, module) pair at increasing trace depth (module-only →
    # +package → full call stack) -- see module docstring / DeepWiki's
    # documented "Trace Levels by Scan Mode". Keep only the deepest
    # (most informative) one per (osv, module) pair.
    best_by_key = {}
    for f in raw_findings:
        trace = f.get("trace") or []
        if not trace:
            continue
        module = trace[0].get("module", "unknown")
        key = (f.get("osv"), module)
        existing = best_by_key.get(key)
        if existing is None or len(trace) > len(existing.get("trace") or []):
            best_by_key[key] = f

    findings = []
    for (osv_id, module), f in best_by_key.items():
        trace = f.get("trace") or [{}]
        first_frame = trace[0]
        osv_entry = osv_entries.get(osv_id, {})

        # More than one frame, or a frame that names a specific
        # function, means govulncheck traced an actual call path to
        # the vulnerable symbol -- not just "this package is imported
        # somewhere". That's the reachability signal worth preserving.
        reachable = len(trace) > 1 or bool(first_frame.get("function"))
        call_trace_summary = None
        if reachable:
            parts = []
            for fr in trace:
                label = fr.get("package") or fr.get("module") or "?"
                if fr.get("function"):
                    label += "." + fr["function"]
                parts.append(label)
            call_trace_summary = " -> ".join(parts)

        description = (osv_entry.get("details") or osv_entry.get("summary") or "No description available")

        findings.append({
            "package": module,
            "installed_version": first_frame.get("version", "unknown"),
            "fix_version": f.get("fixed_version"),
            "vulns": [{"id": osv_id, "description": description[:500]}],
            "ecosystem": "Go",
            "reachable": reachable,
            "call_trace_summary": call_trace_summary,
        })

    return {"success": True, "findings": findings, "total": len(findings),
            "requirements_found": True, "requirements_file": go_mod_path, "error": None}


# ---------------------------------------------------------------------------
# Java: osv-scanner wrapper
# ---------------------------------------------------------------------------
JAVA_MANIFEST_FILENAMES = ["pom.xml", "build.gradle", "build.gradle.kts"]


def _find_java_manifest(repo_path: str) -> str:
    for subdir in COMMON_SUBDIRS:
        for filename in JAVA_MANIFEST_FILENAMES:
            candidate = os.path.join(repo_path, subdir, filename) if subdir else os.path.join(repo_path, filename)
            if os.path.exists(candidate):
                return candidate
    return None


def run_java_dependency_scan(repo_path: str) -> dict:
    """Runs osv-scanner against a Java project's pom.xml / build.gradle.
    Unlike govulncheck, osv-scanner parses the manifest directly and
    does NOT need a Maven/Gradle build to succeed first -- see module
    docstring for why this matters for arbitrary cloned repos.

    Returns the same shape as run_dependency_scan/run_go_dependency_scan,
    tagged "ecosystem": "Maven".
    """
    manifest = _find_java_manifest(repo_path)
    if not manifest:
        return {"success": True, "findings": [], "total": 0,
                "requirements_found": False, "requirements_file": None, "error": None}

    manifest_dir = os.path.dirname(manifest) or repo_path

    try:
        result = subprocess.run(
            ["osv-scanner", "scan", "source", "--format", "json", "--recursive", manifest_dir],
            capture_output=True, text=True, timeout=OSV_SCANNER_TIMEOUT_SECONDS,
            # Explicit UTF-8 -- see run_go_dependency_scan's comment
            # for the real incident that made this necessary on
            # Windows (text=True's locale-default encoding there is
            # cp1252, not UTF-8).
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return {"success": False, "findings": [], "total": 0,
                "requirements_found": True, "requirements_file": manifest,
                "error": "osv-scanner is not installed. See https://github.com/google/osv-scanner/releases"}
    except subprocess.TimeoutExpired:
        return {"success": False, "findings": [], "total": 0,
                "requirements_found": True, "requirements_file": manifest,
                "error": f"osv-scanner timed out after {OSV_SCANNER_TIMEOUT_SECONDS} seconds"}

    output = (result.stdout or "").strip()
    if not output:
        # osv-scanner also exits non-zero (1) when it finds
        # vulnerabilities, as a CI convention -- same reasoning as
        # govulncheck's exit-3 handling above: empty stdout at an
        # unexpected code is a real failure, not "nothing found".
        if result.returncode not in (0, 1):
            return {"success": False, "findings": [], "total": 0,
                    "requirements_found": True, "requirements_file": manifest,
                    "error": f"osv-scanner failed (exit {result.returncode}): {result.stderr.strip()[:500]}"}
        return {"success": True, "findings": [], "total": 0,
                "requirements_found": True, "requirements_file": manifest, "error": None}

    try:
        raw = json.loads(output)
    except json.JSONDecodeError as e:
        return {"success": False, "findings": [], "total": 0,
                "requirements_found": True, "requirements_file": manifest,
                "error": f"Failed to parse osv-scanner JSON output: {e}"}

    findings = []
    for result_entry in raw.get("results", []):
        for pkg_entry in result_entry.get("packages", []):
            # osv-scanner's JSON has used both "package" and "Package"
            # as the key across versions/formats -- check both rather
            # than assuming and silently getting {} on a version
            # mismatch.
            pkg = pkg_entry.get("package") or pkg_entry.get("Package") or {}
            vulns = pkg_entry.get("vulnerabilities", [])
            if not vulns:
                continue

            parsed_vulns = []
            fix_version = None
            for v in vulns:
                parsed_vulns.append({
                    "id": v.get("id", "unknown"),
                    "description": (v.get("summary") or v.get("details") or "No description available")[:500],
                })
                # Fix version lives in affected[].ranges[].events[] as
                # {"fixed": "<version>"} per the OSV schema -- not a
                # top-level field, so this has to walk into it.
                for affected in v.get("affected", []):
                    for rng in affected.get("ranges", []):
                        for event in rng.get("events", []):
                            if fix_version is None and "fixed" in event:
                                fix_version = event["fixed"]

            findings.append({
                "package": pkg.get("name", "unknown"),
                "installed_version": pkg.get("version", "unknown"),
                "fix_version": fix_version,
                "vulns": parsed_vulns,
                "ecosystem": "Maven",
            })

    return {"success": True, "findings": findings, "total": len(findings),
            "requirements_found": True, "requirements_file": manifest, "error": None}