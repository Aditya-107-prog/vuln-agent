"""
tools/fix_generator.py
-----------------------
Step 1 of PR generation: produces proposed fixes WITHOUT touching GitHub.

- Code findings (bandit): sent to the LLM per-file, which returns a
  full corrected file. Written to a local proposed_fixes/ folder for
  review -- nothing is committed or pushed yet.
- Dependency findings: NO LLM needed. OSV.dev already tells us the
  exact fix_version, so this is a deterministic requirements.txt bump.

HUGGING FACE REVISION PINNING (generalized, not file-specific):
Before asking the LLM to fix a file, we scan it for
`from_pretrained("some/model")` calls and look up each model's REAL
current commit hash via the public Hugging Face Hub API. That hash is
handed to the LLM as a verified fact, with an explicit instruction not
to use "main" or invent a hash. "main" is a moving branch name, not a
pin -- using it as the "revision" doesn't fix the underlying issue.
This works for any repo/model, not just this one, since the model name
is extracted from the code being fixed rather than hardcoded.

VERIFICATION LAYERS (this is what actually proves a fix works, rather
than just looking plausible to an LLM). These run in two cost tiers:

Cheap, per-attempt (run on every retry, up to 3x per file):
1. Syntax check (ast.parse) -- catches broken/truncated output for free,
   before even bothering with the critic.
2. pyflakes check -- catches real correctness bugs that syntax-validity
   and bandit both miss: undefined names, broken references, etc.
   Previously the pipeline only checked "is this valid Python" and "is
   this bandit-clean" -- neither of those catches a fix that parses
   fine and looks security-clean but references a variable that
   doesn't exist and would crash the moment the function actually
   runs. Only pyflakes' "undefined name"-class messages are treated as
   real bugs (critical, forces fail); everything else pyflakes reports
   (unused imports, etc.) is cosmetic and tagged moderate, same as any
   other non-blocking critic note.
3. Bandit re-scan (ground truth) -- the fixed code is written to a temp
   file and re-scanned with bandit in isolation. If the SAME bandit rule
   ID that we were trying to fix still fires, that OVERRIDES whatever
   the critic said. The critic is an LLM opinion; bandit re-scanning the
   actual patched code is a fact. Bandit wins on disagreement.
4. Critic review (AWS Bedrock) -- unchanged, still runs, but is now the
   tiebreaker only when bandit is clean, not the final word.

Expensive, run ONCE on the final best candidate only (not per attempt,
to avoid tripling cost across up to 3 retries):
5. Isolated import check -- none of the checks above actually EXECUTE
   the fixed code. A file can be syntactically valid, pyflakes-clean,
   bandit-clean, and critic-approved, while still raising at import
   time (e.g. a name that only breaks once Python actually resolves
   module-level execution, or a decorator/import ordering issue static
   analysis doesn't catch). This check copies the whole repo to a temp
   directory, swaps in the fixed file, and attempts to actually import
   it via runpy -- with run_name deliberately set to something other
   than "__main__" so `if __name__ == "__main__": app.run(...)`-style
   blocks don't actually launch anything, we only want to catch
   import-time errors, not run the app.
6. Existing test suite -- if the repo already has its own tests
   (discovered via pytest), run them against a copy of the repo with
   the fix swapped in. If the repo has none, this is skipped and noted
   as "no tests found" -- absence of tests is not treated as a failure,
   since most target repos (including our test repo) have none.

Both of the expensive checks are opt-in participants in the same
verdict-override logic bandit already uses: a real failure here forces
verdict="fail" regardless of what the critic said, since actually not
running is a stronger signal than any LLM's opinion.

SEVERITY-AWARE PROBLEMS (added after a real incident where a fix left a
JWT verification bypass in place, the critic correctly wrote that down
as a problem, but still scored it 8/10 -- which our old code read as
"pass" because it only looked at the score. Now every problem we
attach to a critique -- whether from the critic itself, from a syntax
failure, or from a bandit-still-flags override -- carries a severity
tag, and critic.py's _normalize_verdict forces "fail" if ANY problem
is tagged "critical", regardless of score.

LangFuse: each code-fix LLM call is logged as a "groq-code-fix"
generation, nested under a "generate_all_fixes" parent trace.
"""

import os
import re
import ast
import shutil
import subprocess
import sys
import tempfile
import time
import contextvars
import importlib.util
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from groq import Groq
from langfuse import observe, get_client, propagate_attributes
from tools.agent_metrics import FIX_ATTEMPTS_PER_FILE, FIX_FINAL_OUTCOME
from tools.critic import critique_fix, CRITIC_ENABLED
from tools.code_scanner import run_code_scan

LANGFUSE_ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"

# How many files to fix concurrently. Each file's fix involves one or more
# Groq calls (fixer) and Bedrock calls (critic), plus subprocess-based
# verification (bandit/pyflakes/pytest) -- all I/O-bound, which is why
# threads (not processes/asyncio) are the right tool here.
#
# Default of 2 (not higher) is deliberate: observed real-world Groq usage
# on this project runs ~5,000-6,000 tokens per fix request, and the
# on-demand tier's default TPM (tokens-per-minute) budget is 12,000 --
# so more than ~2 concurrent requests reliably bursts past that limit
# and triggers 429s (confirmed live: a 4-worker run hit a TPM 429 on the
# very first batch). Raise this only if your Groq tier's TPM limit is
# known to be higher.
FIX_GENERATION_MAX_WORKERS = int(os.environ.get("FIX_GENERATION_MAX_WORKERS", "2"))

# Matches any quoted "org/model"-shaped string anywhere in the file --
# not just literals passed directly to from_pretrained(). This matters
# because the model name is often stored in a variable or default
# parameter (e.g. `model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"`) and
# only referenced by variable name at the actual from_pretrained() call
# site. Scanning the whole file catches that case. False positives are
# harmless: if a matched string isn't a real HF model, the Hub lookup
# below just fails and we skip it silently.
HF_FROM_PRETRAINED_RE = re.compile(r"""["']([A-Za-z0-9][\w\-\.]*/[\w\-\.]+)["']""")

# pyflakes reports lines like:
#   some/file.py:12:5 undefined name 'foo'
#   some/file.py:3:1 'os' imported but unused
# We only treat "undefined name" (and a couple of similarly fatal
# classes) as real bugs that should block a fix. Everything else is
# cosmetic noise, not correctness-breaking.
PYFLAKES_CRITICAL_PATTERNS = (
    "undefined name",
    "undefined local variable",
    "may be undefined, or defined from star imports",
)

TEST_TIMEOUT_SECONDS = 60
IMPORT_CHECK_TIMEOUT_SECONDS = 30
PYTEST_INSTALL_TIMEOUT_SECONDS = 120

# Cache so we only ever check/install pytest ONCE per process, not once
# per file or per retry attempt -- _try_run_tests() runs pytest as a
# subprocess using sys.executable, so as long as pytest exists in THAT
# environment, every sandbox copy (which reuses sys.executable, not its
# own venv) picks it up for free. None = not yet checked, True/False =
# cached result of whether pytest is available to use.
_pytest_available = None


def _ensure_pytest_installed() -> bool:
    """Checks whether `sys.executable -m pytest` works; if not, installs
    pytest into that same environment once. Returns True if pytest is
    (now) available, False if the install itself failed -- in which case
    callers should treat "no tests found/run" as a real "can't check"
    condition rather than silently pretending everything passed.

    Deliberately does NOT install the target repo's own requirements.txt
    -- that's a separate, much heavier and riskier step (arbitrary
    third-party packages, potential version conflicts with our own
    tooling) which is out of scope here. This only guarantees the test
    RUNNER itself is present; a target repo whose tests need packages
    beyond pytest may still hit a collection error naming that specific
    missing package, which is a distinct condition from "pytest isn't
    installed at all" and shows correctly as such in the report.
    """
    global _pytest_available
    if _pytest_available is not None:
        return _pytest_available

    if importlib.util.find_spec("pytest") is not None:
        _pytest_available = True
        return True

    _log("[verify] pytest not found in the verification environment -- installing once (pip install pytest)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "pytest"],
            capture_output=True, text=True, timeout=PYTEST_INSTALL_TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and importlib.util.find_spec("pytest") is not None:
            _log("[verify] pytest installed successfully -- existing-test checks will now run for real.")
            _pytest_available = True
        else:
            _log_error(f"[verify] pytest install failed (exit {result.returncode}): {(result.stderr or '').strip()[:500]}")
            _pytest_available = False
    except Exception as e:
        _log_error(f"[verify] pytest install errored: {e}")
        _pytest_available = False

    return _pytest_available


from logging_config import get_logger

logger = get_logger(__name__)


def _log(msg):
    logger.info(msg)


def _log_error(msg):
    """Always recorded, even in AGENT_QUIET mode -- console verbosity is
    controlled centrally by logging_config's console handler level, not
    by skipping the call here. Used ONLY for actual failures (exceptions,
    API errors) -- never for routine progress logs. AGENT_QUIET should
    hide noise, not hide errors: a swallowed Groq/Bedrock exception in
    the retry loop should never be silently indistinguishable from a
    normal successful run."""
    logger.error(msg)


# ---------------------------------------------------------------------------
# Hugging Face Hub lookup (generalized -- not tied to any specific model)
# ---------------------------------------------------------------------------
def _extract_hf_model_names(file_content: str) -> list:
    """Find every "org/model"-shaped quoted string in the file and return
    the unique candidates. Scans the whole file (not just from_pretrained()
    call sites) since the actual string literal is often assigned to a
    variable/default-parameter and only referenced by name at the call
    site. Works for any model, any repo -- nothing hardcoded."""
    matches = HF_FROM_PRETRAINED_RE.findall(file_content)
    seen = []
    for m in matches:
        if "://" in m or m.count("/") != 1:
            continue  # skip URLs and multi-segment paths, not model IDs
        if m not in seen:
            seen.append(m)
    return seen


def _fetch_hf_commit_hash(model_name: str) -> str:
    """Looks up the REAL current commit hash for a public HF Hub model.
    Returns None if the model isn't found, is private, or the API is
    unreachable -- callers must handle that gracefully."""
    try:
        resp = requests.get(f"https://huggingface.co/api/models/{model_name}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            sha = data.get("sha")
            if sha:
                _log(f"[fix_generator] Verified HF commit hash for {model_name}: {sha}")
                return sha
        _log(f"[fix_generator] Could not verify HF commit hash for {model_name} (status {resp.status_code})")
    except Exception as e:
        _log(f"[fix_generator] HF Hub lookup failed for {model_name}: {e}")
    return None


def _build_hf_revision_facts(file_content: str) -> dict:
    """Returns {model_name: commit_hash} for every from_pretrained() model
    found in the file that we could successfully verify on HF Hub."""
    model_names = _extract_hf_model_names(file_content)
    facts = {}
    for name in model_names:
        sha = _fetch_hf_commit_hash(name)
        if sha:
            facts[name] = sha
    return facts


# ---------------------------------------------------------------------------
# Cheap, per-attempt verification: syntax + correctness + ground-truth rescan
# ---------------------------------------------------------------------------
_EXTENSION_TO_LANGUAGE = {".py": "python", ".go": "go", ".java": "java"}


def _infer_language(relative_path: str) -> str:
    """Same mapping as code_scanner.py's _infer_single_file_language --
    duplicated here rather than imported to avoid a circular import
    (code_scanner doesn't import fix_generator, but keeping this
    self-contained means fix_generator's verification logic doesn't
    silently break if code_scanner's internal helpers ever move).
    Defaults to "python" for unrecognized extensions, preserving every
    existing call site's behavior from before Go/Java support existed.
    """
    _, ext = os.path.splitext(relative_path)
    return _EXTENSION_TO_LANGUAGE.get(ext.lower(), "python")


def _check_syntax(fixed_code: str, language: str = "python"):
    """Returns None if fixed_code is syntactically valid for `language`,
    or an error message string if not. Cheap, deterministic, and
    catches truncated/malformed LLM output before wasting a critic call.

    Python: ast.parse (grammar-only, same as before Go/Java existed).
    Go: gofmt -- not a full compiler check (no type/reference checking,
    same limitation as ast.parse for Python), but genuinely validates
    Go grammar without needing a resolvable module/build context, which
    matters because a single fixed file often can't be built in
    isolation (imports from sibling files in the same package).
    Java: javac dry-compile to a throwaway output dir. Unlike Go,
    Java's compiler needs a class context, so classpath/symbol-
    resolution errors (cannot find symbol, package does not exist) are
    EXPECTED for an isolated single file and are filtered out --  only
    genuine syntax errors (unexpected token, illegal start of
    expression, reached end of file while parsing, etc.) are treated as
    a real failure. This is a known, documented v1 limitation: a
    single-file check can't fully validate Java the way a real project
    build would.
    """
    if language == "python":
        try:
            ast.parse(fixed_code)
            return None
        except SyntaxError as e:
            return f"SyntaxError: {e.msg} (line {e.lineno})"

    if language == "go":
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False, encoding="utf-8") as tmp:
                tmp.write(fixed_code)
                tmp_path = tmp.name
            result = subprocess.run(
                ["gofmt", "-l", tmp_path],
                capture_output=True, text=True, timeout=20,
                encoding="utf-8", errors="replace",
            )
            # gofmt -l prints the filename to stdout ONLY for
            # ill-formatted OR syntactically invalid files; empty
            # stdout AND exit 0 means valid & already formatted.
            # Non-zero exit with stderr content means a genuine parse
            # error (gofmt itself failed to parse the file).
            if result.returncode != 0:
                return f"Go syntax error (gofmt): {(result.stderr or '').strip()[:300]}"
            return None
        except FileNotFoundError:
            _log("[verify] gofmt not found (part of the Go toolchain) -- skipping Go syntax check for this attempt")
            return None
        except subprocess.TimeoutExpired:
            return "gofmt timed out"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if language == "java":
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="vuln_agent_javac_")
            # javac cares about filename matching the public class name
            # -- best-effort extraction; falls back to a generic name
            # if no public class is found (e.g. package-private class,
            # or the fix is a fragment), in which case javac will
            # reliably fail with "class X is public, should be
            # declared in a file named X.java" which we then need to
            # filter out too (it's a naming artifact of this check, not
            # a real problem with the fix).
            class_match = re.search(r'public\s+(?:final\s+|abstract\s+)?class\s+(\w+)', fixed_code)
            file_name = f"{class_match.group(1)}.java" if class_match else "Fix.java"
            java_path = os.path.join(tmp_dir, file_name)
            with open(java_path, "w", encoding="utf-8") as f:
                f.write(fixed_code)

            result = subprocess.run(
                ["javac", "-d", tmp_dir, java_path],
                capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                return None

            stderr = result.stderr or ""
            # Filter out classpath/symbol-resolution noise that's
            # EXPECTED for a single isolated file with no project
            # context -- only surface lines that indicate a genuine
            # grammar/syntax problem.
            benign_patterns = ["cannot find symbol", "package", "does not exist", "class, interface, or enum expected"]
            real_errors = [
                line for line in stderr.splitlines()
                if "error:" in line and not any(p in line for p in benign_patterns)
            ]
            if real_errors:
                return f"Java syntax error (javac): {'; '.join(real_errors)[:300]}"
            return None  # only benign classpath-context errors -- treat as syntactically valid
        except FileNotFoundError:
            _log("[verify] javac not found (part of the JDK) -- skipping Java syntax check for this attempt")
            return None
        except subprocess.TimeoutExpired:
            return "javac timed out"
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except OSError:
                    pass

    # Unknown language -- nothing to check, don't block the pipeline
    return None


def _top_level_def_names(code: str, language: str = "python") -> set:
    """Returns the set of top-level function/class names defined in a
    file. Used by _check_no_definitions_deleted below.

    Python: exact, via ast (unchanged from before Go/Java support).
    Go/Java: regex-based heuristic, NOT an AST-perfect parse -- matches
    top-level `func Name(...)` / `type Name ...` for Go, and method/
    class declarations for Java. This is intentionally the same
    "regex-fallback" approach flagged as a known limitation back when
    Go/Java scanning support was first planned: a proper extractor
    would need go/ast (Go) or javalang (Java), which is more setup
    than this check's payoff currently justifies. Good enough to catch
    the exact regression class this check exists for (an LLM silently
    deleting a whole function while "fixing" something else) without
    needing a second language toolchain integration.
    """
    if language == "python":
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    if language == "go":
        # Top-level (column 0, not indented) func/type declarations.
        # `(?m)^func\s+(?:\([^)]*\)\s+)?(\w+)` handles both plain
        # functions and methods with a receiver, e.g.
        # `func (s *Server) Handle(...)`.
        names = set()
        for m in re.finditer(r'(?m)^func\s+(?:\([^)]*\)\s+)?(\w+)', code):
            names.add(m.group(1))
        for m in re.finditer(r'(?m)^type\s+(\w+)', code):
            names.add(m.group(1))
        return names

    if language == "java":
        # Method-level heuristic: modifiers + return type + name + "(".
        # Deliberately loose (doesn't distinguish class-body nesting
        # depth) -- false positives here just mean a rename inside a
        # nested/anonymous class could be mis-flagged as a "deletion",
        # which fails safe (extra manual review) rather than silently
        # missing a real one.
        names = set()
        for m in re.finditer(r'(?m)^\s*(?:public|private|protected)\s+(?:static\s+|final\s+|abstract\s+)*[\w<>\[\],\s]+?\s+(\w+)\s*\(', code):
            names.add(m.group(1))
        for m in re.finditer(r'(?m)^\s*(?:public|private|protected)?\s*(?:static\s+|final\s+|abstract\s+)*class\s+(\w+)', code):
            names.add(m.group(1))
        return names

    return set()


def _check_no_definitions_deleted(original_code: str, fixed_code: str, language: str = "python") -> dict:
    """Deterministic ground-truth check: does the fixed code still
    define every top-level function/class the original file did?

    This exists because an LLM critic can be wrong -- we saw a real
    case where a fix deleted an entire function (get_user_safe) while
    ALSO reintroducing every original vulnerability, and the critic
    still scored it 10/10 with zero problems listed, despite being
    given both the original and fixed code and being explicitly
    instructed to check for silently broken functionality. Rather than
    depend on the critic reliably noticing this every time, check it
    directly (via ast for Python, regex heuristics for Go/Java -- see
    _top_level_def_names) -- the same "don't trust the LLM's judgment
    for something we can verify deterministically" philosophy already
    used for the ground-truth re-scan and correctness check elsewhere
    in this file.

    Returns {"deleted": [names removed], "ok": bool}. Intentionally
    only flags REMOVALS, not additions/renames -- a fix legitimately
    adding a new helper function is fine and shouldn't be penalized.
    """
    original_names = _top_level_def_names(original_code, language)
    fixed_names = _top_level_def_names(fixed_code, language)
    deleted = sorted(original_names - fixed_names)
    return {"deleted": deleted, "ok": len(deleted) == 0}


def _check_rule_coverage(original_test_ids: set, critique: dict) -> dict:
    """Deterministic cross-check: does the critic's own
    rule_ids_addressed list actually cover every bandit rule ID this
    fix was supposed to resolve?

    This exists because a "pass" verdict from the critic only proves
    the critic didn't THINK of a reason to object -- it doesn't prove
    the critic actually engaged with every original finding. Nothing
    before this checked whether the critic's writeup even mentioned
    all of them; a critic that silently ignored one of two findings
    and praised the one it did look at could still produce a clean
    "pass". Same "don't trust an LLM's silence as confirmation"
    philosophy as _check_no_definitions_deleted, applied to critic
    coverage instead of code structure.

    Unlike bandit-still-flags or deleted-definitions (which are
    ground-truth PROOF of a regression and force a hard "fail"), a
    coverage gap only proves the critic's write-up is incomplete -- the
    fix itself might still be fine. So the caller downgrades rather
    than force-fails on this one; see the call site.

    Returns {"missing": [rule_ids not addressed], "ok": bool,
    "applicable": bool}. applicable=False means there was nothing
    meaningful to check (no rule IDs to begin with, or the critic was
    unavailable) -- distinct from ok=True, so callers/report can show
    "N/A" rather than a potentially misleading "covered".
    """
    if not original_test_ids:
        return {"missing": [], "ok": True, "applicable": False}
    if not critique or critique.get("verdict") == "unavailable":
        return {"missing": [], "ok": True, "applicable": False}

    addressed = {str(r) for r in (critique.get("rule_ids_addressed") or [])}
    missing = sorted(str(r) for r in original_test_ids if str(r) not in addressed)
    return {"missing": missing, "ok": len(missing) == 0, "applicable": True}


def _check_pyflakes(fixed_code: str, relative_path: str, language: str = "python") -> dict:
    """Runs pyflakes on the fixed code in isolation and classifies its
    findings. Returns {"has_critical": bool, "problems": [{"text","severity"}]}.

    Only "undefined name"-class messages are treated as real bugs -- the
    kind of thing that would actually crash the code at runtime, which
    neither ast.parse() (checks grammar, not references) nor bandit
    (checks security patterns, not correctness) catches. Everything else
    pyflakes reports (unused imports, redefinition, etc.) is noted as a
    moderate/cosmetic problem, consistent with how bandit/critic problems
    are already severity-tagged elsewhere in this pipeline.

    Go/Java: DELIBERATELY SKIPPED for now (documented v1 gap, same as
    flagged when Go/Java support was first planned). The equivalents
    would be `go vet`/`staticcheck` (Go) and `checkstyle` (Java), but
    both need real project/classpath context to run meaningfully on a
    single isolated file the way pyflakes doesn't -- adding them
    without that context would produce mostly-noise findings, which is
    worse than no check at all for a verification layer whose whole
    point is trustworthy signal. Coverage for Go/Java currently comes
    from: syntax check (gofmt/javac), ground-truth re-scan (semgrep),
    deleted-definition check, and the critic -- one fewer layer than
    Python gets, which is worth knowing rather than pretending parity.
    """
    if language != "python":
        return {"has_critical": False, "problems": [], "skipped": True,
                "skip_reason": f"correctness check not yet implemented for {language} (see docstring)"}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(fixed_code)
            tmp_path = tmp.name

        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", tmp_path],
            capture_output=True, text=True, timeout=20,
            encoding="utf-8", errors="replace",
        )

        problems = []
        has_critical = False
        for line in (result.stdout or "").splitlines():
            # Format: <path>:<line>:<col>: <message>  (col is optional
            # depending on pyflakes version)
            parts = line.split(":", 3)
            message = parts[-1].strip() if len(parts) >= 2 else line.strip()
            if not message:
                continue
            is_critical = any(pat in message.lower() for pat in PYFLAKES_CRITICAL_PATTERNS)
            severity = "critical" if is_critical else "moderate"
            if is_critical:
                has_critical = True
            problems.append({"text": f"pyflakes: {message}", "severity": severity})

        return {"has_critical": has_critical, "problems": problems}

    except FileNotFoundError:
        _log("[verify] pyflakes is not installed -- skipping this check. Run: pip install pyflakes")
        return {"has_critical": False, "problems": [], "error": "pyflakes not installed"}
    except subprocess.TimeoutExpired:
        _log(f"[verify] pyflakes timed out for {relative_path} -- skipping this check")
        return {"has_critical": False, "problems": [], "error": "pyflakes timed out"}
    except Exception as e:
        _log(f"[verify] pyflakes errored for {relative_path}: {e}")
        return {"has_critical": False, "problems": [], "error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _rescan_with_bandit(fixed_code: str, relative_path: str, original_test_ids: set, language: str = "python") -> dict:
    """Writes fixed_code to a temp file (correct extension for
    `language`) and re-runs the SAME scanner family that produced the
    original findings (bandit for Python, semgrep for Go/Java -- see
    code_scanner.py's run_code_scan, which already dispatches by
    extension) on it ALONE, as a ground-truth check on whether the
    original vulnerability (by rule ID) is still present after the
    fix. This does not trust the critic's or the fixer's own claim
    that the issue is resolved -- it re-detects the same way the
    original scan did. (Function name kept as _rescan_with_bandit for
    minimal diff against every existing call site; it's no longer
    bandit-specific -- see run_code_scan's language dispatch.)

    Returns {"still_vulnerable": bool, "remaining_issues": [rule_ids], "error": optional str}
    """
    if not original_test_ids:
        return {"still_vulnerable": False, "remaining_issues": []}

    suffix = {"python": ".py", "go": ".go", "java": ".java"}.get(language, ".py")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as tmp:
            tmp.write(fixed_code)
            tmp_path = tmp.name

        scan_result = run_code_scan(tmp_path, recursive=False, language=language)
        if not scan_result["success"]:
            _log(f"[verify] ground-truth re-scan failed for {relative_path}: {scan_result.get('error')} -- skipping ground-truth check for this attempt")
            return {"still_vulnerable": False, "remaining_issues": [], "error": scan_result.get("error")}

        found_ids = {f["issue"] for f in scan_result["findings"]}
        remaining = sorted(found_ids & original_test_ids)
        return {"still_vulnerable": bool(remaining), "remaining_issues": remaining}

    except Exception as e:
        _log(f"[verify] ground-truth re-scan errored for {relative_path}: {e}")
        return {"still_vulnerable": False, "remaining_issues": [], "error": str(e)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Expensive, once-per-file verification: isolated import + existing tests
# ---------------------------------------------------------------------------
def _make_repo_copy_with_fix(repo_path: str, relative_path: str, fixed_code: str):
    """Copies the whole repo to a temp directory and overwrites
    relative_path with fixed_code, so that sibling-module imports inside
    the fixed file resolve correctly (unlike the single-file temp-file
    approach used for bandit/pyflakes, which only works because those
    tools don't need the rest of the repo to be present).

    Returns the temp directory path. Caller is responsible for cleanup
    (shutil.rmtree). Returns None if the copy fails.
    """
    try:
        tmp_dir = tempfile.mkdtemp(prefix="vuln_agent_verify_")
        # Skip heavy/irrelevant directories to keep the copy fast and
        # avoid copying .git history unnecessarily.
        shutil.copytree(
            repo_path, tmp_dir, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv", "venv", "node_modules"),
        )
        target_path = os.path.join(tmp_dir, relative_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(fixed_code)
        return tmp_dir
    except Exception as e:
        _log(f"[verify] Failed to create isolated repo copy for import/test check: {e}")
        return None


def _try_isolated_import(repo_path: str, relative_path: str, code: str) -> dict:
    """Actually attempts to import the given code (either the original
    file or a proposed fix) as a real Python module, inside a full copy
    of the repo (so its own sibling imports resolve), rather than just
    checking its text is grammatically valid. This is the one check in
    the whole pipeline that executes code at all -- everything else
    (syntax, pyflakes, bandit, critic) only reads it.

    Generic on purpose: used for BOTH the original file (as a baseline)
    and the fixed file, so we can tell the difference between "this
    fails because our environment doesn't have the target repo's
    dependencies installed" (also true of the original, unmodified
    file -- not the fix's fault) and "this fails because the fix
    genuinely broke something that used to work" (a real regression).
    See _check_isolated_import_with_baseline() for that comparison.

    run_name is deliberately set to something other than "__main__" so
    that `if __name__ == "__main__": app.run(...)`-style blocks do NOT
    actually execute -- we want to catch import-time errors (missing
    names, broken imports, decorator issues), not launch the target
    app.

    Returns {"import_ok": bool, "error": optional str}.
    """
    tmp_dir = _make_repo_copy_with_fix(repo_path, relative_path, code)
    if tmp_dir is None:
        return {"import_ok": True, "error": "skipped -- could not create isolated copy", "skipped": True}

    try:
        target_path = os.path.join(tmp_dir, relative_path)
        script = (
            "import runpy, sys\n"
            "try:\n"
            f"    runpy.run_path(r'{target_path}', run_name='not_main')\n"
            "except Exception as e:\n"
            "    print(f'IMPORT_CHECK_FAILED: {type(e).__name__}: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_dir, capture_output=True, text=True,
            timeout=IMPORT_CHECK_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            error_line = next(
                (l for l in result.stderr.splitlines() if "IMPORT_CHECK_FAILED" in l),
                (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown import error"),
            )
            return {"import_ok": False, "error": error_line}
        return {"import_ok": True, "error": None}

    except subprocess.TimeoutExpired:
        return {"import_ok": True, "error": "timed out (inconclusive, not counted as failure)", "skipped": True}
    except Exception as e:
        return {"import_ok": True, "error": str(e), "skipped": True}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _error_class(error_text: str) -> str:
    """Extracts a rough 'error class' from an import/test failure message
    for baseline comparison -- e.g. 'ModuleNotFoundError' from
    "IMPORT_CHECK_FAILED: ModuleNotFoundError: No module named 'flask_sqlalchemy'".
    Used to decide whether the fixed file's failure is the SAME kind of
    failure the original file already had (environment gap, not the
    fix's fault) or a genuinely different/new failure (real regression).
    """
    if not error_text:
        return ""
    match = re.search(r"([A-Za-z]+Error)", error_text)
    return match.group(1) if match else error_text.strip()[:40]


def _check_isolated_import_with_baseline(repo_path: str, relative_path: str, original_content: str, fixed_code: str) -> dict:
    """Runs the isolated import check on the FIXED file, but first
    establishes a baseline by running the identical check against the
    ORIGINAL, unmodified file in the same isolated environment.

    Why: our verification sandbox does not have the target repo's own
    dependencies installed (e.g. flask_sqlalchemy, zapv2) -- installing
    them would add real time/risk for uncertain benefit, and even then
    wouldn't guarantee our environment matches the repo's intended one.
    Without a baseline, a repo that simply can't run in our sandbox at
    all would cause EVERY fix to false-fail this check, regardless of
    whether the fix itself is any good -- which is exactly what happened
    the first time this ran (ModuleNotFoundError on both files, purely
    because those packages aren't installed here).

    Comparing against the original file distinguishes:
    - baseline also fails, SAME error class -> environment gap, not the
      fix's fault. Check is skipped, verdict is not affected.
    - baseline succeeds, fixed fails -> genuine regression introduced by
      the fix. Counted as a real failure.
    - baseline fails with a DIFFERENT error class than the fix -> still
      treated as inconclusive/skipped rather than blocking, since we
      can't cleanly attribute this to the fix either.
    """
    baseline = _try_isolated_import(repo_path, relative_path, original_content)
    fixed = _try_isolated_import(repo_path, relative_path, fixed_code)

    if fixed.get("skipped"):
        return {**fixed, "baseline": baseline}

    if fixed["import_ok"]:
        return {"import_ok": True, "error": None, "baseline": baseline}

    if not baseline.get("skipped") and not baseline["import_ok"]:
        baseline_class = _error_class(baseline.get("error", ""))
        fixed_class = _error_class(fixed.get("error", ""))
        if baseline_class == fixed_class:
            _log(f"[verify] Isolated import check for {relative_path}: fixed file fails with the same error class ({fixed_class}) as the ORIGINAL file -- this is an environment gap (missing dependency), not something the fix caused. Skipping this check.")
            return {"import_ok": True, "error": f"skipped -- original file has the same failure ({fixed_class}), not attributable to the fix", "skipped": True, "baseline": baseline}

    _log(f"[verify] Isolated import FAILED for {relative_path} (original file imports fine, but the fix does not): {fixed.get('error')}")
    return {"import_ok": False, "error": fixed.get("error"), "baseline": baseline}


def _try_run_tests(repo_path: str, relative_path: str, code: str) -> dict:
    """If the repo has its own pytest-discoverable tests, runs them
    against a copy of the repo with the given code (either original or
    fixed) swapped in. Absence of tests is NOT a failure -- most target
    repos (including our test repo, which has zero automated tests per
    the handoff) won't have any.

    Generic on purpose -- used for both a baseline run (original file)
    and the real check (fixed file); see
    _run_existing_tests_with_baseline() for the comparison logic.

    Returns {"tests_found": bool, "tests_passed": bool or None, "summary": str}
    """
    if not _ensure_pytest_installed():
        return {"tests_found": False, "tests_passed": None, "summary": "skipped -- pytest is not installed and the automatic install failed"}

    tmp_dir = _make_repo_copy_with_fix(repo_path, relative_path, code)
    if tmp_dir is None:
        return {"tests_found": False, "tests_passed": None, "summary": "skipped -- could not create isolated copy"}

    try:
        collect = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=tmp_dir, capture_output=True, text=True, timeout=20,
        )
        collected_output = (collect.stdout or "") + (collect.stderr or "")

        # Use pytest's actual exit code rather than scraping stdout text,
        # which is fragile and version-dependent (a previous version of
        # this check missed the "no tests ran" wording that appears when
        # a repo has zero test files at all, and misreported that as a
        # test FAILURE rather than "no tests exist"). Pytest's own exit
        # codes are stable and documented:
        #   0 = all collected tests passed
        #   1 = some tests failed
        #   2 = execution interrupted (often a collection-time error,
        #       e.g. an ImportError while importing a test module)
        #   3 = internal pytest error
        #   4 = usage error (bad CLI args -- shouldn't happen here)
        #   5 = no tests were collected at all
        if collect.returncode == 5:
            return {"tests_found": False, "tests_passed": None, "summary": "no tests found in repo"}

        if collect.returncode not in (0, 5):
            # Something went wrong just COLLECTING tests -- most commonly
            # an import error in a test file (e.g. a missing dependency
            # our sandbox doesn't have installed). This is exactly the
            # class of failure the baseline comparison in
            # _run_existing_tests_with_baseline() is designed to catch
            # and not blame on the fix.
            # Prefer pytest's own traceback lines (conventionally prefixed
            # "E   ") which contain the actual exception, e.g.
            # "E   ModuleNotFoundError: No module named 'foo'". A naive
            # "any line containing 'error'" search matches pytest's own
            # "=========== ERRORS ===========" section banner first,
            # since it appears earlier in the output than the real
            # exception -- which produces a useless message like
            # "test collection errored: === ERRORS ===" instead of
            # naming the actual missing module/import problem.
            error_line = next(
                (l.strip() for l in collected_output.splitlines() if l.strip().startswith("E ") and "error" in l.lower()),
                None,
            )
            if error_line is None:
                # Fallback: any line naming a real exception class
                # (WordError: ...), skipping pure "===" banner lines.
                error_line = next(
                    (l.strip() for l in collected_output.splitlines()
                     if "error" in l.lower() and not set(l.strip()) <= {"=", " "}),
                    f"test collection exited with code {collect.returncode}",
                )
            return {"tests_found": False, "tests_passed": None, "summary": f"test collection errored: {error_line}", "collection_error": True}

        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=tmp_dir, capture_output=True, text=True, timeout=TEST_TIMEOUT_SECONDS,
        )

        if run.returncode == 5:
            # Defensive: collect-only said tests exist but the real run
            # found none anyway (can happen with certain plugins/markers
            # filtering everything out). Treat the same as "no tests".
            return {"tests_found": False, "tests_passed": None, "summary": "no tests found in repo"}

        passed = run.returncode == 0
        summary_line = next(
            (l for l in reversed((run.stdout or "").splitlines()) if l.strip()),
            "(no summary line captured)",
        )
        return {"tests_found": True, "tests_passed": passed, "summary": summary_line}

    except subprocess.TimeoutExpired:
        return {"tests_found": True, "tests_passed": None, "summary": "timed out (inconclusive, not counted as failure)"}
    except FileNotFoundError:
        return {"tests_found": False, "tests_passed": None, "summary": "pytest not installed"}
    except Exception as e:
        return {"tests_found": False, "tests_passed": None, "summary": f"error: {e}"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_existing_tests_with_baseline(repo_path: str, relative_path: str, original_content: str, fixed_code: str) -> dict:
    """Runs the existing-tests check on the FIXED file, but first
    establishes a baseline against the ORIGINAL file, for the same
    reason as _check_isolated_import_with_baseline(): if the target
    repo's tests can't even be collected/run in our sandbox because a
    dependency isn't installed, that's true of the original file too --
    it's not something the fix caused, and blaming the fix for it would
    be a false failure (this is exactly what happened when this check
    first ran: pytest collection errored on a missing module for BOTH
    the original and fixed file, which we hadn't yet distinguished).

    - baseline also can't collect/run (same reason) -> environment gap,
      skip this check, don't affect verdict.
    - baseline runs and passes, fixed fails -> genuine regression.
    - baseline has no tests at all -> unaffected either way, "no tests
      found" as before.
    """
    baseline = _try_run_tests(repo_path, relative_path, original_content)
    fixed = _try_run_tests(repo_path, relative_path, fixed_code)

    if not fixed.get("tests_found"):
        # No tests at all, or fixed run couldn't even collect -- check
        # if baseline had the same problem before treating this as a
        # real failure.
        if fixed.get("collection_error") and baseline.get("collection_error"):
            _log(f"[verify] Existing-test check for {relative_path}: test collection fails on the ORIGINAL file too (same environment gap) -- marking Inconclusive, not attributable to the fix.")
            return {"tests_found": True, "tests_passed": None,
                    "summary": f"inconclusive -- {fixed.get('summary')} (original file has the same collection error, not attributable to the fix)",
                    "baseline": baseline}
        return {**fixed, "baseline": baseline}

    if fixed.get("tests_passed") is not False:
        return {**fixed, "baseline": baseline}

    # fixed tests_passed is False -- check whether baseline also failed
    # for the same underlying reason before blaming the fix.
    if baseline.get("collection_error"):
        _log(f"[verify] Existing-test check for {relative_path}: baseline (original file) can't even collect tests -- marking Inconclusive, not attributable to the fix.")
        return {"tests_found": True, "tests_passed": None,
                "summary": "inconclusive -- original file's tests can't be collected either, not attributable to the fix",
                "baseline": baseline}

    _log(f"[verify] Existing test suite FAILED for {relative_path} after fix applied (baseline passes): {fixed.get('summary')}")
    return {**fixed, "baseline": baseline}


# Rank used to pick the "best" attempt across retries. A syntax-invalid
# attempt can NEVER outrank a syntax-valid one, no matter its score.
# "unavailable" (critic couldn't run) ranks like needs_improvement so a
# bandit-clean, critic-unavailable attempt still beats nothing at all,
# but a real "pass" always wins.
_VERDICT_RANK = {"pass": 2, "needs_improvement": 1, "fail": 0, "unavailable": 1}


def _candidate_rank(candidate: dict) -> tuple:
    """Ranks candidates so the objectively best one wins, NOT just the
    one the critic happened to score highest. Ground truth (bandit's
    remaining-issue count) is checked BEFORE the critic score, since
    the critic can be wrong -- this is the same reason bandit is
    allowed to force-override the critic's verdict elsewhere in this
    file. Without this, two "fail" candidates with different numbers
    of actual remaining vulnerabilities could get tie-broken purely on
    critic score, letting an objectively worse candidate (e.g. one
    that reintroduces ALL the original findings, or even deletes a
    function) win over a genuinely better one just because the critic
    over-scored it.
    """
    if not candidate["verification"]["syntax_valid"]:
        return (-1, 0, 0, -1)
    verdict = candidate["verdict"]
    score = candidate.get("score") if candidate.get("score") is not None else 0
    remaining = candidate["verification"].get("remaining_issues") or []
    deleted = candidate["verification"].get("deleted_definitions") or []
    # Fewer remaining bandit issues AND fewer deleted definitions are
    # both better -- negate so higher tuple values still mean "better"
    # (consistent with score/verdict both being "higher is better").
    return (_VERDICT_RANK.get(verdict, 0), -len(remaining), -len(deleted), score)


# ---------------------------------------------------------------------------
# Code fixes (LLM-based, per file)
# ---------------------------------------------------------------------------
_LANGUAGE_DISPLAY_NAME = {"python": "Python", "go": "Go", "java": "Java"}
_LANGUAGE_IDIOM_NOTES = {
    "python": "",
    "go": """
GO-SPECIFIC NOTES:
- Follow standard Go error handling (`if err != nil { ... }`) -- do not
  swallow errors silently or introduce a panic where the original code
  returned an error.
- Preserve exported (capitalized) function/type names exactly -- other
  files in this package may depend on them; renaming an exported
  symbol is a breaking change, not a security fix.
- Keep formatting gofmt-compatible (tabs for indentation, standard
  brace placement) -- the fix will be checked with `gofmt`.
""",
    "java": """
JAVA-SPECIFIC NOTES:
- Use try-with-resources for anything that implements Closeable/
  AutoCloseable (Connection, Statement, ResultSet, etc.) rather than
  manual close() calls, unless the original code's resource-management
  pattern must be preserved for another reason.
- For SQL, use PreparedStatement with parameter binding (`?`
  placeholders + setX(...) calls), never string concatenation into
  the query text.
- Preserve the class's public method signatures exactly -- other code
  in this codebase may call these methods; changing a signature is a
  breaking change, not a security fix.
""",
}


def _build_code_fix_prompt(relative_path: str, file_content: str, findings: list, hf_revision_facts: dict, critique_feedback: dict = None, language: str = "python") -> str:
    findings_desc = "\n".join(
        f"- Line {f.get('line', '?')} [{f.get('severity', '?')}]: {f.get('description', '')}"
        for f in findings
    )

    hf_facts_block = ""
    if hf_revision_facts:
        facts_lines = "\n".join(
            f'- For `{model}`, the VERIFIED current commit hash is `{sha}`. '
            f'Use exactly `revision="{sha}"`.'
            for model, sha in hf_revision_facts.items()
        )
        hf_facts_block = f"""
VERIFIED FACTS (use these exactly, do not substitute your own values):
{facts_lines}

IMPORTANT: `revision="main"` is NOT a valid fix for unpinned model loading.
"main" is a moving branch name, not a pin -- it changes over time just like
having no revision at all. You MUST use the verified commit hash above.
"""

    retry_block = ""
    if critique_feedback:
        problems = critique_feedback.get("problems", [])
        problems_text = "\n".join(
            f"- [{p.get('severity', 'moderate').upper()}] {p.get('text', p) if isinstance(p, dict) else p}"
            for p in problems
        )
        retry_block = f"""
YOUR PREVIOUS ATTEMPT WAS REVIEWED AND REJECTED by an independent reviewer.
Problems found:
{problems_text}

Reviewer's suggestion: {critique_feedback.get("suggestions", "")}

Fix these specific problems in your new attempt, especially any marked
CRITICAL -- those are non-negotiable. Do not repeat the same mistake.
"""

    display_name = _LANGUAGE_DISPLAY_NAME.get(language, language)
    idiom_notes = _LANGUAGE_IDIOM_NOTES.get(language, "")
    # Code fence language tag -- affects nothing functionally, but a
    # wrong tag (e.g. "python" for a Go file) is a strong signal to a
    # human reviewer glancing at the report that something upstream is
    # misconfigured, so it's worth getting right even though the model
    # doesn't strictly need it.
    fence_lang = {"python": "python", "go": "go", "java": "java"}.get(language, "")

    return f"""You are a senior application security engineer fixing vulnerabilities in a {display_name} file.

File: {relative_path}

Findings to fix:
{findings_desc}
{hf_facts_block}{retry_block}
Current file content:
```{fence_lang}
{file_content}
```

Fix ONLY the specific issues listed above. Do not refactor unrelated code,
change formatting, or rename anything not required for the fix. Preserve
all existing functionality.

CRITICAL BEHAVIOR-PRESERVATION REQUIREMENT: this repository has its own
existing test suite, and your fix will be run against it after this. Your
fix MUST continue to pass every test that passes today -- do not change a
function's return type, return value for valid/safe inputs, exceptions
raised, or any other observable behavior, unless the vulnerability itself
requires it. In particular:
- If replacing a dangerous function (e.g. eval) with a safer one, make sure
  the replacement still supports the SAME range of valid inputs and
  produces the SAME kind of output the original did -- a safer function
  that can no longer do what the original did for legitimate use is not
  a correct fix, it just moves the bug from "insecure" to "broken."
- If changing an algorithm (e.g. a hash function), only change what's
  necessary to remove the vulnerability -- do not incidentally change
  the format/type/length of the output unless that's specifically what
  the vulnerability required.
{idiom_notes}
Respond in EXACTLY this format, nothing else:

===EXPLANATION===
<1-3 sentences explaining what you changed and why>
===FIXED_CODE===
<the complete corrected file content, and nothing after it>
"""


def _call_groq_for_code_fix(prompt: str, relative_path: str, attempt: int = None, scan_id: str = None) -> str:
    client = Groq()

    def _do_call():
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise security-focused code-fixing assistant. Follow the requested output format exactly, and always use verified facts given to you instead of guessing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        return response

    if not LANGFUSE_ENABLED:
        return _do_call().choices[0].message.content

    langfuse = get_client()
    # metadata carries attempt + scan_id so a trace can be filtered/read
    # without guessing from generation order or timestamps alone --
    # previously only {"file": relative_path} was recorded, which made
    # a 3-attempt retry sequence for the same file indistinguishable
    # except by which one happened to come first in the UI.
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="groq-code-fix",
        model=MODEL_NAME,
        input=prompt,
        metadata={"file": relative_path, "attempt": attempt, "scan_id": scan_id},
    ) as generation:
        response = _do_call()
        usage = response.usage
        generation.update(
            output=response.choices[0].message.content,
            usage_details={
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
            } if usage else None,
        )
    return response.choices[0].message.content


def _parse_fix_response(raw: str) -> dict:
    explanation = ""
    fixed_code = raw

    if "===EXPLANATION===" in raw and "===FIXED_CODE===" in raw:
        try:
            after_marker = raw.split("===EXPLANATION===", 1)[1]
            explanation, fixed_code = after_marker.split("===FIXED_CODE===", 1)
            explanation = explanation.strip()
            fixed_code = fixed_code.strip()
        except Exception:
            pass

    # Strip accidental markdown fences if the model added them anyway --
    # generic pattern now (```python, ```go, ```java, or no tag at all),
    # not hardcoded to python only.
    fixed_code = re.sub(r"^```\w*\n?", "", fixed_code)
    fixed_code = re.sub(r"\n?```$", "", fixed_code)

    return {"explanation": explanation or "No explanation provided.", "fixed_code": fixed_code}


def _verify_no_unpinned_revisions_remain(fixed_code: str, hf_revision_facts: dict) -> list:
    """Post-hoc sanity check: if we gave the LLM verified hashes, make sure
    it actually used them and didn't fall back to revision="main" anyway.
    Returns a list of warning strings (empty if all clear)."""
    warnings = []
    if not hf_revision_facts:
        return warnings

    if re.search(r'revision\s*=\s*["\']main["\']', fixed_code):
        warnings.append(
            'Fixed code still contains revision="main" despite being given a '
            'verified commit hash -- the LLM likely ignored the instruction. '
            'Manual review required.'
        )

    for model, sha in hf_revision_facts.items():
        if model in fixed_code and sha not in fixed_code:
            warnings.append(
                f'Verified hash for {model} ({sha}) does not appear in the fixed '
                f'code -- the pin may be missing or wrong. Manual review required.'
            )

    return warnings


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    """Groq's 429 error messages include a human-readable hint like
    'Please try again in 13.045s' (or '1m2.5s' for longer waits). Parse
    that out so we can actually wait the suggested time before retrying,
    instead of retrying instantly into the same still-exhausted budget.
    Returns None if this isn't a rate-limit error or no hint was found.
    """
    message = str(exc)
    if "rate_limit" not in message and "429" not in message:
        return None

    # Matches "13.045s" or "1m2.5s" style durations after "try again in"
    match = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", message, re.IGNORECASE)
    if not match:
        return None

    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    # Small buffer on top of Groq's own estimate, capped so a single
    # retry never blocks a file for an unreasonably long time.
    return min(minutes * 60 + seconds + 0.5, 60.0)


def _generate_fix_for_one_file(repo_path: str, relative_path: str, findings: list, scan_id: str = None) -> Optional[dict]:
    """Drafts + verifies a fix for a single file, through the full
    retry/verification pipeline (layers 1-6). Returns a result dict on
    success, or None if this file should be skipped (bad path, missing
    file, read error, or no usable fix produced after all retries).

    Pulled out of generate_code_fixes() as its own function so it can be
    run concurrently across files via ThreadPoolExecutor -- each file's
    fix is fully independent of every other file's (own LLM calls, own
    verification, own output file), so there's no shared state to worry
    about between concurrent calls other than logging interleaving,
    which is harmless (just means log lines from different files can
    interleave in the terminal).
    """
    if not relative_path or relative_path.startswith(".."):
        _log(f"[fix_generator] Skipping suspicious path outside repo: {relative_path}")
        return None

    full_path = os.path.join(repo_path, relative_path)
    if not os.path.exists(full_path):
        _log(f"[fix_generator] Skipping {relative_path} -- file not found on disk")
        return None

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            original_content = f.read()
    except Exception as e:
        _log(f"[fix_generator] Skipping {relative_path} -- read error: {e}")
        return None

    _log(f"[fix_generator] Generating fix for {relative_path} ({len(findings)} finding(s))...")

    # Inferred ONCE per file from the extension and threaded through
    # every verification layer below (syntax check, correctness check,
    # ground-truth rescan, deleted-def check, prompt) -- see
    # _infer_language's docstring for the extension mapping and the
    # "default to python" fallback for unrecognized extensions.
    language = _infer_language(relative_path)

    # Look up any Hugging Face models referenced in this file BEFORE
    # asking the LLM to fix anything, so we can hand it real facts.
    # (HF model-loading findings are currently a Python-specific
    # concern -- this is harmless to still run for Go/Java files since
    # it will simply find no matches and return an empty dict.)
    hf_revision_facts = _build_hf_revision_facts(original_content)
    finding_descs = [f.get("description", "") for f in findings]
    original_test_ids = {f.get("issue") for f in findings if f.get("issue")}

    MAX_RETRIES = 2  # up to 3 total attempts (1 initial + 2 retries)
    best_candidate = None
    critique_feedback = None
    attempt = 0

    while attempt <= MAX_RETRIES:
        attempt += 1
        try:
            prompt = _build_code_fix_prompt(
                relative_path, original_content, findings, hf_revision_facts,
                critique_feedback=critique_feedback, language=language,
            )
            raw_response = _call_groq_for_code_fix(prompt, relative_path, attempt=attempt, scan_id=scan_id)
            parsed = _parse_fix_response(raw_response)
        except Exception as e:
            wait_seconds = _extract_retry_after_seconds(e)
            if wait_seconds is not None:
                _log_error(
                    f"[fix_generator] Rate limited generating fix for {relative_path} "
                    f"(attempt {attempt}/{MAX_RETRIES + 1}) -- waiting {wait_seconds:.1f}s before retrying: {e}"
                )
                time.sleep(wait_seconds)
            else:
                _log_error(f"[fix_generator] LLM fix failed for {relative_path} (attempt {attempt}/{MAX_RETRIES + 1}): {e}")
            if best_candidate is not None:
                break
            continue

        # --- Layer 1: syntax check (cheap, deterministic, checked first) ---
        syntax_error = _check_syntax(parsed["fixed_code"], language=language)

        if syntax_error:
            _log(f"[verify] {relative_path} attempt {attempt}: syntax check FAILED -- {syntax_error}")
            verification = {"syntax_valid": False, "bandit_still_flags": None, "remaining_issues": [], "pyflakes": None}
            critique = {
                "score": 0,
                "verdict": "fail",
                "problems": [{"text": f"Fixed code is not valid Python: {syntax_error}", "severity": "critical"}],
                "suggestions": "Produce syntactically valid Python. Do not truncate the file or leave partial edits.",
            }
            candidate = {"parsed": parsed, "critique": critique, "verification": verification, "verdict": "fail", "score": 0}

        else:
            # --- Layer 2: pyflakes check (catches real correctness bugs
            # that syntax-validity and bandit both miss -- e.g. a
            # reference to a name that doesn't exist) ---
            pyflakes_result = _check_pyflakes(parsed["fixed_code"], relative_path, language=language)
            if pyflakes_result["has_critical"]:
                _log(f"[verify] {relative_path} attempt {attempt}: pyflakes found a real bug (undefined name or similar)")

            # --- Layer 4: critic review (runs before the bandit
            # override below, so we have its opinion to merge with
            # both bandit's and pyflakes' ground truth) ---
            critique = None
            if CRITIC_ENABLED:
                critique = critique_fix(relative_path, finding_descs, original_content, parsed["fixed_code"], parsed["explanation"], finding_rule_ids=sorted(original_test_ids), attempt=attempt, scan_id=scan_id)
                if critique.get("verdict") == "unavailable":
                    _log(f"[critic] Unavailable for {relative_path}: {critique.get('error')}")
                else:
                    _log(f"[critic] {relative_path} attempt {attempt} scored {critique.get('score')}/10 ({critique.get('verdict')})")

            # --- Layer 3: bandit ground-truth re-scan (overrides critic) ---
            bandit_result = _rescan_with_bandit(parsed["fixed_code"], relative_path, original_test_ids, language=language)
            deletion_result = _check_no_definitions_deleted(original_content, parsed["fixed_code"], language=language)
            coverage_result = _check_rule_coverage(original_test_ids, critique)
            verification = {
                "syntax_valid": True,
                "bandit_still_flags": bandit_result["still_vulnerable"],
                "remaining_issues": bandit_result["remaining_issues"],
                "pyflakes": pyflakes_result,
                "deleted_definitions": deletion_result["deleted"],
                "rule_coverage": coverage_result,
            }

            # Merge in pyflakes' real-bug findings the same way bandit's
            # ground-truth override works: a genuine correctness bug
            # forces fail regardless of what the critic or bandit said,
            # since a fix that crashes at runtime is not a fix.
            extra_critical_problems = []
            force_fail = False

            if bandit_result["still_vulnerable"]:
                _log(f"[verify] {relative_path} attempt {attempt}: bandit STILL detects {bandit_result['remaining_issues']} -- overriding verdict to fail regardless of critic")
                extra_critical_problems += [
                    {"text": f"bandit still flags rule {rid} in the fixed code -- the vulnerability was not actually resolved, only cosmetically changed", "severity": "critical"}
                    for rid in bandit_result["remaining_issues"]
                ]
                force_fail = True

            if not deletion_result["ok"]:
                _log(f"[verify] {relative_path} attempt {attempt}: fix DELETED existing function/class definition(s) {deletion_result['deleted']} -- overriding verdict to fail regardless of critic")
                extra_critical_problems += [
                    {"text": f"the fix removed the existing definition of '{name}', which was present in the original file -- this is a regression, not a fix, even if the critic didn't flag it", "severity": "critical"}
                    for name in deletion_result["deleted"]
                ]
                force_fail = True

            if pyflakes_result["has_critical"]:
                extra_critical_problems += [p for p in pyflakes_result["problems"] if p["severity"] == "critical"]
                force_fail = True
            elif pyflakes_result["problems"]:
                # Non-blocking pyflakes notes (unused imports etc.) are
                # still worth surfacing in the summary, same as any
                # other moderate problem.
                extra_critical_problems += pyflakes_result["problems"]

            if force_fail:
                if critique and critique.get("verdict") != "unavailable":
                    critique = {**critique, "problems": critique.get("problems", []) + extra_critical_problems, "verdict": "fail"}
                else:
                    critique = {
                        "score": critique.get("score") if critique else None,
                        "verdict": "fail",
                        "problems": extra_critical_problems,
                        "suggestions": "Ensure the actual pattern bandit/pyflakes flags is fully resolved, not just superficially altered.",
                    }
                verdict = "fail"
            else:
                if extra_critical_problems and critique and critique.get("verdict") != "unavailable":
                    critique = {**critique, "problems": critique.get("problems", []) + extra_critical_problems}
                verdict = critique.get("verdict") if (critique and critique.get("verdict") != "unavailable") else "pass"

                # Rule-ID coverage downgrade: NOT a hard fail like bandit/
                # deletion above, since a coverage gap only proves the
                # critic's writeup is incomplete, not that the fix is
                # actually broken (bandit's ground-truth re-scan above
                # already cleared it, or we wouldn't be in this branch).
                # A "pass" the critic never fully engaged with shouldn't
                # be trusted as a real pass, but it also shouldn't be
                # punished as harshly as a proven regression -- so this
                # only ever downgrades pass -> needs_improvement, never
                # forces fail and never upgrades anything.
                if not coverage_result["ok"] and verdict == "pass":
                    _log(f"[verify] {relative_path} attempt {attempt}: critic gave 'pass' but rule_ids_addressed is missing {coverage_result['missing']} -- downgrading to needs_improvement (critic did not confirm every original finding was resolved)")
                    coverage_problem = {
                        "text": f"critic gave an overall pass but its rule_ids_addressed list did not confirm rule(s) {coverage_result['missing']} were actually resolved -- treat as unconfirmed, not verified",
                        "severity": "moderate",
                    }
                    critique = {**critique, "problems": critique.get("problems", []) + [coverage_problem], "verdict": "needs_improvement"}
                    verdict = "needs_improvement"

            score = critique.get("score") if (critique and critique.get("score") is not None) else (10 if verdict == "pass" else 0)
            candidate = {"parsed": parsed, "critique": critique, "verification": verification, "verdict": verdict, "score": score}

        if best_candidate is None or _candidate_rank(candidate) > _candidate_rank(best_candidate):
            best_candidate = candidate

        if candidate["verdict"] == "pass":
            break

        if attempt <= MAX_RETRIES:
            critique_feedback = candidate["critique"]
            _log(f"[fix_generator] Retrying {relative_path} with feedback from attempt {attempt}...")

    if best_candidate is None:
        _log(f"[fix_generator] No usable fix produced for {relative_path} after {attempt} attempt(s) -- skipping")
        # Recorded even on total failure -- "how many files get given up
        # on entirely" is exactly the kind of thing an agent-health
        # dashboard needs to surface, not just successful fixes.
        FIX_ATTEMPTS_PER_FILE.observe(attempt)
        FIX_FINAL_OUTCOME.labels(verdict="no_usable_fix").inc()
        return None

    parsed = best_candidate["parsed"]
    critique = best_candidate["critique"]
    verification = best_candidate["verification"]

    # --- Layer 5: isolated import check (expensive, once on the final
    # best candidate). Still gated on the cheap-layer verdict not
    # already being "fail" -- this check answers "does the fix even
    # run", which is moot to spend cost on once bandit/pyflakes have
    # already rejected the candidate for an unrelated reason.
    #
    # PYTHON ONLY: this literally does `importlib` on the file, which
    # is a Python-specific mechanism -- there's no equivalent "just
    # import it" concept for Go (needs `go build`) or Java (needs
    # `javac`+`java` with a real classpath), and both of those are
    # heavier operations than this check is designed for. Skipped
    # cleanly for Go/Java rather than attempted and silently
    # misinterpreted -- same documented v1 gap as the pyflakes-
    # equivalent check above. ---
    if language != "python":
        import_result = {"import_ok": True, "error": None, "skipped": True,
                          "skip_reason": f"isolated import check is Python-specific, not yet implemented for {language}"}
    elif verification["syntax_valid"] and best_candidate["verdict"] != "fail":
        import_result = _check_isolated_import_with_baseline(repo_path, relative_path, original_content, parsed["fixed_code"])
    else:
        import_result = {"import_ok": True, "error": None, "skipped": True}
    verification["import_check"] = import_result

    # --- Layer 6: existing-test check (expensive, once on the final
    # best candidate). Unlike the import check above, this runs
    # whenever the code is at least syntactically valid, REGARDLESS of
    # the bandit/critic verdict -- so the test column always shows a
    # real Pass/Fail/Inconclusive answer, even for a candidate bandit
    # already flagged for an unrelated reason (e.g. a file that still
    # trips a bandit rule but we still want to know independently
    # whether it breaks the repo's own tests).
    #
    # PYTHON ONLY: this runs pytest, which obviously can't run Go/Java
    # tests. `go test` and `mvn test`/`gradle test` are real equivalents
    # but need a working build first (see code_scanner.py's docstring
    # on why this pipeline generally avoids requiring one) -- adding
    # them is future work, not something to fake by silently reporting
    # "no tests found" as if that were the same as "tests pass". ---
    if language != "python":
        test_result = {"tests_found": False, "tests_passed": None, "summary": None, "skipped": True,
                        "skip_reason": f"existing-test check runs pytest, not yet implemented for {language} (would need go test / mvn test)"}
    elif verification["syntax_valid"]:
        test_result = _run_existing_tests_with_baseline(repo_path, relative_path, original_content, parsed["fixed_code"])

        # A genuine regression -- baseline passes, fixed fails, not an
        # environment gap -- gets retried with the failure fed back to
        # the fix-generator LLM, using the SAME retry budget as the
        # bandit-failure loop above (MAX_RETRIES), rather than a single
        # one-shot attempt. Only applies if this candidate wasn't
        # already a bandit/pyflakes "fail" for an unrelated reason (in
        # that case bandit's fail already dominates the verdict, so a
        # retry here can't change the outcome).
        is_genuine_regression = (
            test_result.get("tests_found")
            and test_result.get("tests_passed") is False
            and not str(test_result.get("summary", "")).startswith("skipped")
        )
        test_retry_attempt = 0
        while is_genuine_regression and best_candidate["verdict"] != "fail" and test_retry_attempt < MAX_RETRIES:
            test_retry_attempt += 1
            _log(f"[verify] {relative_path}: fix passes bandit/critic but FAILS the repo's existing test suite (baseline passes) -- retrying ({test_retry_attempt}/{MAX_RETRIES}) with test-failure feedback: {test_result.get('summary')}")
            test_retry_feedback = {
                "score": best_candidate.get("score"),
                "verdict": "needs_improvement",
                "problems": [{
                    "text": f"The fix passes the security scan but breaks the repo's own existing test suite, which passes on the original code. Test failure: {test_result.get('summary')}",
                    "severity": "critical",
                }],
                "suggestions": "Fix the code so it resolves the vulnerability WITHOUT breaking the existing passing tests. Do not reintroduce the original vulnerability while fixing this.",
            }
            try:
                retry_prompt = _build_code_fix_prompt(
                    relative_path, original_content, findings, hf_revision_facts,
                    critique_feedback=test_retry_feedback, language=language,
                )
                retry_raw = _call_groq_for_code_fix(retry_prompt, relative_path)
                retry_parsed = _parse_fix_response(retry_raw)
                retry_syntax_error = _check_syntax(retry_parsed["fixed_code"], language=language)

                if retry_syntax_error:
                    _log(f"[verify] {relative_path}: test-failure retry {test_retry_attempt}/{MAX_RETRIES} produced invalid syntax -- discarding, keeping best candidate so far.")
                    continue

                # Re-run the cheap ground-truth layers on the new
                # candidate -- a test-focused retry could reintroduce
                # the original vulnerability, so we can't just trust
                # it still passes what the earlier attempt passed.
                retry_bandit = _rescan_with_bandit(retry_parsed["fixed_code"], relative_path, original_test_ids, language=language)
                retry_pyflakes = _check_pyflakes(retry_parsed["fixed_code"], relative_path, language=language)
                retry_deletion = _check_no_definitions_deleted(original_content, retry_parsed["fixed_code"], language=language)
                retry_clean = (not retry_bandit["still_vulnerable"]) and (not retry_pyflakes["has_critical"]) and retry_deletion["ok"]

                if not retry_clean:
                    if not retry_deletion["ok"]:
                        _log(f"[verify] {relative_path}: test-failure retry {test_retry_attempt}/{MAX_RETRIES} DELETED existing definition(s) {retry_deletion['deleted']} -- discarding, keeping best candidate so far.")
                    else:
                        _log(f"[verify] {relative_path}: test-failure retry {test_retry_attempt}/{MAX_RETRIES} reintroduced a vulnerability or real bug -- discarding, keeping best candidate so far.")
                    continue

                retry_test_result = _run_existing_tests_with_baseline(repo_path, relative_path, original_content, retry_parsed["fixed_code"])
                if retry_test_result.get("tests_passed") is False:
                    # This attempt didn't fix it (or made it worse) --
                    # discard it and keep BOTH the current best
                    # candidate's code AND its test_result unchanged,
                    # so the report accurately describes the code
                    # that's actually being kept, not a discarded
                    # attempt's numbers. Loop continues to the next
                    # attempt if budget remains.
                    _log(f"[verify] {relative_path}: test-failure retry {test_retry_attempt}/{MAX_RETRIES} still fails the existing test suite ({retry_test_result.get('summary')}) -- discarding, keeping best candidate so far.")
                else:
                    _log(f"[verify] {relative_path}: test-failure retry {test_retry_attempt}/{MAX_RETRIES} succeeded -- fix no longer breaks the existing test suite.")
                    parsed = retry_parsed
                    test_result = retry_test_result
                    verification["bandit_still_flags"] = retry_bandit["still_vulnerable"]
                    verification["remaining_issues"] = retry_bandit["remaining_issues"]
                    verification["pyflakes"] = retry_pyflakes
                    if CRITIC_ENABLED:
                        retry_critique = critique_fix(relative_path, finding_descs, original_content, retry_parsed["fixed_code"], retry_parsed["explanation"], finding_rule_ids=sorted(original_test_ids))
                        if retry_critique.get("verdict") != "unavailable":
                            # Same coverage cross-check as the main loop --
                            # a test-failure retry that gets a "pass" from
                            # the critic still needs its rule_ids_addressed
                            # list checked, or this path would silently
                            # bypass the very check the main loop applies.
                            retry_coverage = _check_rule_coverage(original_test_ids, retry_critique)
                            if not retry_coverage["ok"] and retry_critique.get("verdict") == "pass":
                                _log(f"[verify] {relative_path}: test-retry critic gave 'pass' but rule_ids_addressed is missing {retry_coverage['missing']} -- downgrading to needs_improvement")
                                retry_critique = {
                                    **retry_critique,
                                    "problems": retry_critique.get("problems", []) + [{
                                        "text": f"critic gave an overall pass but its rule_ids_addressed list did not confirm rule(s) {retry_coverage['missing']} were actually resolved -- treat as unconfirmed, not verified",
                                        "severity": "moderate",
                                    }],
                                    "verdict": "needs_improvement",
                                }
                            verification["rule_coverage"] = retry_coverage
                            critique = retry_critique
                    break  # success -- stop retrying
            except Exception as e:
                _log_error(f"[fix_generator] test-failure retry {test_retry_attempt}/{MAX_RETRIES} errored for {relative_path}: {e}")
    else:
        test_result = {"tests_found": False, "tests_passed": None, "summary": "skipped -- fixed code failed syntax validation"}
    verification["existing_tests"] = test_result

    runtime_problems = []
    runtime_force_fail = False

    if not import_result.get("import_ok", True):
        runtime_problems.append({
            "text": f"Fixed file fails to import/run: {import_result.get('error')}",
            "severity": "critical",
        })
        runtime_force_fail = True

    if test_result.get("tests_found") and test_result.get("tests_passed") is False:
        runtime_problems.append({
            "text": f"Repo's existing test suite fails against this fix: {test_result.get('summary')}",
            "severity": "critical",
        })
        runtime_force_fail = True

    if runtime_force_fail:
        _log(f"[verify] {relative_path}: runtime check(s) failed -- overriding verdict to fail regardless of syntax/bandit/pyflakes/critic")
        if critique and critique.get("verdict") != "unavailable":
            critique = {**critique, "problems": critique.get("problems", []) + runtime_problems, "verdict": "fail"}
        else:
            critique = {
                "score": critique.get("score") if critique else None,
                "verdict": "fail",
                "problems": runtime_problems,
                "suggestions": "The fix must actually run without error and must not break the repo's existing tests.",
            }
        best_candidate["critique"] = critique
        best_candidate["verdict"] = "fail"

    best_candidate["parsed"] = parsed

    if best_candidate["verdict"] != "pass":
        if not verification["syntax_valid"]:
            _log(f"[fix_generator] {relative_path}: all attempts produced invalid syntax -- keeping best attempt for manual review, NOT confirmed fixed.")
        elif verification["bandit_still_flags"]:
            _log(f"[fix_generator] {relative_path}: bandit still flags {verification['remaining_issues']} after {attempt} attempt(s) -- keeping best-scoring version for manual review, NOT confirmed fixed.")
        elif not verification.get("import_check", {}).get("import_ok", True):
            _log(f"[fix_generator] {relative_path}: fix does not actually run -- keeping best-scoring version for manual review, NOT confirmed fixed.")
        elif verification.get("existing_tests", {}).get("tests_passed") is False:
            _log(f"[fix_generator] {relative_path}: fix breaks the repo's existing tests -- keeping best-scoring version for manual review, NOT confirmed fixed.")
        else:
            _log(f"[critic] {relative_path} still not passing after {attempt} attempt(s) -- keeping best scoring version for manual review.")

    warnings = _verify_no_unpinned_revisions_remain(parsed["fixed_code"], hf_revision_facts)
    for w in warnings:
        _log(f"[fix_generator] WARNING for {relative_path}: {w}")

    if not verification["syntax_valid"]:
        warnings.append("Fixed code FAILED Python syntax validation -- this file is NOT safe to merge as-is.")
    elif verification["bandit_still_flags"]:
        warnings.append(f"bandit re-scan shows the original vulnerability rule(s) {verification['remaining_issues']} still present after the fix -- this file is NOT confirmed resolved.")
    if not verification.get("import_check", {}).get("import_ok", True):
        warnings.append(f"Fixed code does not actually run: {verification['import_check'].get('error')}")
    if verification.get("existing_tests", {}).get("tests_passed") is False:
        warnings.append(f"Fixed code breaks the repo's existing test suite: {verification['existing_tests'].get('summary')}")

    # Recorded HERE and not earlier -- best_candidate["verdict"] can
    # still flip to "fail" via runtime_force_fail (import check / test
    # suite, layers 5-6) well after the main retry loop picked a best
    # candidate, so this is the first point where the verdict is
    # actually final. attempt count includes only the main retry loop's
    # attempts, not the separate test-failure retry sub-loop above --
    # that's an intentional simplification (the two loops have
    # different budgets/triggers), fine for a "how many drafts did the
    # LLM need" signal without needing a second histogram just for it.
    FIX_ATTEMPTS_PER_FILE.observe(attempt)
    FIX_FINAL_OUTCOME.labels(verdict=best_candidate["verdict"]).inc()

    return {
        "relative_path": relative_path,
        "original_path": full_path,
        # original_code (not just original_path) is required here, not
        # optional: original_path points into the temp clone directory
        # (github_fetcher's tempdir), which is deleted once the scan
        # finishes -- without the actual text preserved in this dict,
        # nothing downstream (e.g. a diff viewer) could ever show what
        # changed after the scan completes, only what the file looks
        # like now.
        "original_code": original_content,
        "findings": findings,
        "parsed": parsed,
        "hf_revision_facts": hf_revision_facts,
        "warnings": warnings,
        "critique": critique,
        "verification": verification,
    }


def generate_code_fixes(repo_path: str, code_findings: list, output_dir: str, scan_id: str = None) -> list:
    if not code_findings:
        return []

    findings_by_file = defaultdict(list)
    for f in code_findings:
        raw_file = f.get("file", "")
        if not raw_file:
            continue
        # Bandit gives an ABSOLUTE path. Normalize to a path relative to
        # repo_path so we never accidentally write outside output_dir,
        # and so this same relative_path can later be used as the path
        # inside the git branch/commit in Step 2.
        try:
            relative_path = os.path.relpath(raw_file, repo_path)
        except ValueError:
            # Can happen on Windows if paths are on different drives
            relative_path = os.path.basename(raw_file)
        findings_by_file[relative_path].append(f)

    # Each file's fix is fully independent (own LLM calls, own
    # verification, own output file) -- so files are processed
    # concurrently via a thread pool. Kept to FIX_GENERATION_MAX_WORKERS
    # at a time (not "all files at once") to avoid bursting past
    # Groq/Bedrock per-minute rate limits on repos with many files.
    # Results are collected keyed by relative_path and re-assembled in
    # the ORIGINAL findings_by_file order afterward, so output ordering
    # (and thus SUMMARY.md/report ordering) stays deterministic
    # regardless of which file's thread happens to finish first.
    #
    # IMPORTANT (Langfuse): this function runs inside an @observe-wrapped
    # call (generate_all_fixes), which sets the "current span" via
    # Python's contextvars in THIS thread. contextvars are NOT
    # automatically inherited by new threads spawned by
    # ThreadPoolExecutor -- each worker thread would otherwise start
    # with an empty context, so every file's groq-code-fix generation
    # would show up in Langfuse as an unrelated root trace instead of
    # nested correctly under this scan's trace.
    #
    # Fix: capture a SEPARATE contextvars.copy_context() snapshot per
    # file (not one shared object!) here in the main thread, where the
    # parent span is active, and run each worker inside its own copy.
    # A single Context object cannot be .run() from two threads at the
    # same time (Python raises "cannot enter context: already entered"),
    # so each file needs its own independent copy -- copy_context()
    # called once per file, all still capturing the same parent span
    # since they're all taken at this same point in the main thread.
    file_items = list(findings_by_file.items())
    results_by_path = {}

    with ThreadPoolExecutor(max_workers=FIX_GENERATION_MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(
                contextvars.copy_context().run, _generate_fix_for_one_file, repo_path, relative_path, findings, scan_id
            ): relative_path
            for relative_path, findings in file_items
        }
        for future in as_completed(future_to_path):
            relative_path = future_to_path[future]
            try:
                outcome = future.result()
            except Exception as e:
                _log_error(f"[fix_generator] Unexpected error generating fix for {relative_path}: {e}")
                outcome = None
            if outcome is not None:
                results_by_path[relative_path] = outcome

    # Write output files and build final result dicts in original order
    # (writing to disk is cheap/fast, kept sequential and outside the
    # thread pool to avoid any concern about concurrent file I/O).
    results = []
    for relative_path, _findings in file_items:
        outcome = results_by_path.get(relative_path)
        if outcome is None:
            continue

        out_path = os.path.join(output_dir, "code", relative_path)
        assert not os.path.isabs(relative_path), f"relative_path must not be absolute: {relative_path}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(outcome["parsed"]["fixed_code"])

        results.append({
            "relative_path": outcome["relative_path"],
            "original_path": outcome["original_path"],
            "fixed_path": out_path,
            # FIX: original_code and fixed_code were never carried
            # through this reshaping step, even after original_code was
            # added to _generate_fix_for_one_file's return dict -- this
            # function builds a brand new dict here and simply never
            # copied either field over. Everything downstream (web_app's
            # diff viewer) that expected the raw per-file dict to flow
            # through unmodified was wrong; THIS is the actual assembled
            # shape nodes.py and web_app.py receive. Confirmed via a
            # real diff-viewer screenshot showing "(no textual
            # differences)" on a fix that very clearly changed
            # real code (critique referenced specific code changes) --
            # both original_code and fixed_code were silently empty
            # strings the whole time, not genuinely identical.
            "original_code": outcome["original_code"],
            "fixed_code": outcome["parsed"]["fixed_code"],
            "findings_addressed": len(outcome["findings"]),
            "explanation": outcome["parsed"]["explanation"],
            "hf_revision_facts": outcome["hf_revision_facts"],
            "warnings": outcome["warnings"],
            "critique": outcome["critique"],
            "verification": outcome["verification"],
            "findings": outcome["findings"],
        })

    return results


# ---------------------------------------------------------------------------
# Dependency fixes (rule-based, no LLM -- OSV.dev already gives the fix version)
# ---------------------------------------------------------------------------
def generate_python_dependency_fixes(repo_path: str, dep_findings: list, output_dir: str, requirements_file_path: str = None) -> list:
    """
    Handles two distinct cases:
    - DIRECT: the vulnerable package already has a line in requirements.txt
      -> bump that line's version in place.
    - TRANSITIVE: the vulnerable package is pulled in indirectly (no line
      of its own) -> since there's nothing to bump, add an explicit pin
      for it. This forces pip to install the patched version regardless
      of what pulled it in, which is the standard fix used by tools like
      Dependabot/pip-tools when the direct parent hasn't yet released a
      version that itself points to the patched sub-dependency.

    requirements_file_path: the ACTUAL path dep_scanner.py found (may be
    in a subfolder like app/requirements.txt, not always repo root). If
    not provided (e.g. called from a standalone test script), falls back
    to assuming repo root for backward compatibility.
    """
    if not dep_findings:
        return []

    req_file = requirements_file_path or os.path.join(repo_path, "requirements.txt")
    if not os.path.exists(req_file):
        _log(f"[fix_generator] No requirements.txt found at {req_file}, skipping dependency fixes")
        return []

    fix_map = {}
    for f in dep_findings:
        pkg = f.get("package", "").lower()
        fix_version = f.get("fix_version")
        if fix_version and fix_version != "no fix available":
            fix_map[pkg] = {"fix_version": fix_version, "osv_ids": f.get("osv_ids", [])}

    if not fix_map:
        _log("[fix_generator] No fix versions available for any vulnerable package")
        return []

    with open(req_file, "r", encoding="utf-8") as f:
        original_lines = f.readlines()

    new_lines = []
    fixes_applied = []
    direct_packages_seen = set()

    for line in original_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue

        pkg_name = re.split(r"[=<>~!\[]", stripped)[0].strip().lower()
        direct_packages_seen.add(pkg_name)

        if pkg_name in fix_map:
            new_version = fix_map[pkg_name]["fix_version"]
            new_line = f"{pkg_name}=={new_version}\n"
            new_lines.append(new_line)
            fixes_applied.append({
                "package": pkg_name,
                "old_line": stripped,
                "new_line": new_line.strip(),
                "fix_type": "direct",
            })
        else:
            new_lines.append(line)

    # Anything in fix_map that was never seen as a direct line is transitive
    # -- add an explicit pin for it rather than trying to bump a line that
    # doesn't exist.
    transitive_additions = []
    for pkg_name, info in fix_map.items():
        if pkg_name not in direct_packages_seen:
            new_version = info["fix_version"]
            osv_note = f" (fixes {', '.join(info['osv_ids'])})" if info.get("osv_ids") else ""
            new_line = f"{pkg_name}=={new_version}  # transitive dependency pin{osv_note}\n"
            transitive_additions.append(new_line)
            fixes_applied.append({
                "package": pkg_name,
                "old_line": "(not previously pinned -- transitive dependency)",
                "new_line": new_line.strip(),
                "fix_type": "transitive",
            })

    if transitive_additions:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append("\n# Transitive dependency pins added by Vulnerability Finder Agent\n")
        new_lines.extend(transitive_additions)

    if not fixes_applied:
        return []

    out_path = os.path.join(output_dir, "requirements.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    requirements_relative_path = os.path.relpath(req_file, repo_path)

    for fix in fixes_applied:
        fix["fixed_path"] = out_path
        fix["requirements_relative_path"] = requirements_relative_path

    return fixes_applied


# ---------------------------------------------------------------------------
# Go dependency fixes -- uses the real Go toolchain (`go get`/`go mod
# tidy`) rather than hand-editing go.mod, unlike the Python path above.
# ---------------------------------------------------------------------------
def generate_go_dependency_fixes(repo_path: str, go_findings: list, go_mod_path: str = None, output_dir: str = None) -> list:
    """`go get module@version` is the correct, toolchain-native way to
    bump a Go dependency -- it updates BOTH go.mod and go.sum together
    (including transitive checksum changes), which a hand-edited
    go.mod alone would leave inconsistent and unbuildable. This is a
    real (and arguably safer) difference from the Python path, which
    can only ever regex-edit requirements.txt text since pip has no
    equivalent single-command "bump and reconcile" operation.

    repo_path is the scan's disposable temp clone (see run_scan_thread/
    github_fetcher.py) -- running `go get` directly in it is safe
    precisely because nothing else depends on that clone surviving
    unmodified afterward, unlike a developer's real working copy.
    """
    if not go_findings:
        return []
    if not go_mod_path:
        _log("[fix_generator] No go.mod path provided, skipping Go dependency fixes")
        return []

    go_dir = os.path.dirname(go_mod_path) or repo_path

    fix_targets = []
    for f in go_findings:
        pkg = f.get("package")
        fix_version = f.get("fix_version")
        if pkg and fix_version and fix_version != "no fix available":
            fix_targets.append((pkg, fix_version, f.get("vulns") or []))

    if not fix_targets:
        _log("[fix_generator] No fix versions available for any vulnerable Go package")
        return []

    fixes_applied = []
    for pkg, fix_version, vulns in fix_targets:
        try:
            result = subprocess.run(
                ["go", "get", f"{pkg}@{fix_version}"],
                cwd=go_dir, capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            _log("[fix_generator] Go toolchain not found -- skipping remaining Go dependency fixes")
            break
        except subprocess.TimeoutExpired:
            _log(f"[fix_generator] `go get {pkg}@{fix_version}` timed out -- skipping this package")
            continue

        if result.returncode != 0:
            # One package failing (e.g. a breaking major-version bump
            # that needs code changes go get can't make) shouldn't stop
            # the others -- log and continue, same non-fatal pattern
            # used throughout this file.
            _log(f"[fix_generator] `go get {pkg}@{fix_version}` failed: {result.stderr.strip()[:300]}")
            continue

        osv_note = f" (fixes {', '.join(v['id'] for v in vulns)})" if vulns else ""
        _log(f"[fix_generator] Bumped Go module {pkg} to {fix_version}{osv_note}")
        fixes_applied.append({
            "package": pkg,
            "old_line": f"{pkg} (previous version)",
            "new_line": f"{pkg} {fix_version}",
            "fix_type": "go_get",
        })

    if not fixes_applied:
        return []

    # go mod tidy reconciles go.sum and drops now-unused indirect
    # entries -- run ONCE after all `go get` calls, not per-package,
    # since each `go get` already leaves the module graph in a valid
    # (if not fully tidied) state and re-running tidy repeatedly is
    # just wasted work.
    try:
        subprocess.run(["go", "mod", "tidy"], cwd=go_dir, capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        _log(f"[fix_generator] `go mod tidy` did not complete cleanly: {e}")

    # Copy the now-modified go.mod (and go.sum, if present) into
    # output_dir -- mirrors the Python path's convention of writing the
    # fixed manifest there, which github_pr.py's existing apply-fix
    # step already knows how to pick up.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename in ("go.mod", "go.sum"):
            src = os.path.join(go_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, filename))

    go_mod_relative_path = os.path.relpath(go_mod_path, repo_path)
    for fix in fixes_applied:
        fix["fixed_path"] = os.path.join(output_dir, "go.mod") if output_dir else go_mod_path
        fix["requirements_relative_path"] = go_mod_relative_path

    return fixes_applied


# ---------------------------------------------------------------------------
# Java dependency fixes -- Maven (pom.xml) only. Gradle (build.gradle /
# build.gradle.kts) findings are reported but not auto-fixed -- see
# docstring below for why.
# ---------------------------------------------------------------------------
def generate_java_dependency_fixes(repo_path: str, java_findings: list, java_manifest_path: str = None, output_dir: str = None) -> list:
    """Only handles pom.xml. build.gradle/build.gradle.kts are Groovy/
    Kotlin DSL, not structured markup -- safely bumping a version
    there needs a real parser for that DSL, which this session doesn't
    have; a regex risks corrupting a build script in ways a person
    then has to debug by hand, which is worse than just not attempting
    it and saying so clearly (which is what happens below).

    Uses a targeted TEXT replacement (not an XML round-trip via
    ElementTree) specifically to avoid reformatting the whole file --
    an ElementTree write-back would silently normalize whitespace,
    attribute quoting, and comments across the ENTIRE pom.xml, turning
    a one-line version bump into a huge, unreviewable diff. A property-
    based version (<version>${some.prop}</version>) is detected and
    skipped rather than corrupted, since bumping the property itself
    could affect other dependencies that share it.
    """
    if not java_findings:
        return []
    if not java_manifest_path:
        _log("[fix_generator] No Java manifest path provided, skipping Java dependency fixes")
        return []

    if not java_manifest_path.lower().endswith("pom.xml"):
        _log(f"[fix_generator] Java dependency auto-fix only supports Maven (pom.xml) right now -- "
             f"skipping {len(java_findings)} Gradle finding(s) at {java_manifest_path}")
        return []

    with open(java_manifest_path, "r", encoding="utf-8") as f:
        pom_content = f.read()

    fixes_applied = []
    for finding in java_findings:
        pkg = finding.get("package", "")
        fix_version = finding.get("fix_version")
        if not fix_version or fix_version == "no fix available":
            continue
        if ":" not in pkg:
            # osv-scanner's Maven package name is always "groupId:artifactId"
            # -- anything else means the finding shape doesn't match
            # what this function expects, so skip rather than guess.
            _log(f"[fix_generator] Unexpected Maven package name format {pkg!r}, skipping")
            continue
        group_id, artifact_id = pkg.split(":", 1)

        # Matches a <dependency> block containing this exact groupId +
        # artifactId (in either order), capturing its <version> tag's
        # inner text specifically -- deliberately tolerant of
        # whitespace/attribute variation between the two tags, since
        # real pom.xml formatting varies a lot between projects.
        pattern = re.compile(
            r"(<dependency>(?:(?!</dependency>).)*?<groupId>\s*" + re.escape(group_id) + r"\s*</groupId>"
            r"(?:(?!</dependency>).)*?<artifactId>\s*" + re.escape(artifact_id) + r"\s*</artifactId>"
            r"(?:(?!</dependency>).)*?<version>\s*)([^<]+?)(\s*</version>)",
            re.DOTALL,
        )
        match = pattern.search(pom_content)
        if not match:
            _log(f"[fix_generator] Could not locate a <version> tag for {pkg} in pom.xml -- skipping (may use a parent-managed version)")
            continue

        old_version = match.group(2).strip()
        if old_version.startswith("${"):
            # Version comes from a Maven property (e.g. a <properties>
            # block or parent POM) -- bumping this one <version> tag
            # wouldn't even do anything, and bumping the property
            # itself could silently affect other dependencies that
            # share it. Flag as needing a manual fix rather than
            # silently no-op'ing or guessing which property to touch.
            _log(f"[fix_generator] {pkg}'s version is set via a Maven property ({old_version}) -- needs a manual fix, skipping")
            continue

        pom_content = pom_content[:match.start()] + match.group(1) + fix_version + match.group(3) + pom_content[match.end():]

        osv_note = f" (fixes {', '.join(v['id'] for v in finding.get('vulns', []))})" if finding.get("vulns") else ""
        _log(f"[fix_generator] Bumped Maven dependency {pkg} {old_version} -> {fix_version}{osv_note}")
        fixes_applied.append({
            "package": pkg,
            "old_line": f"<version>{old_version}</version>",
            "new_line": f"<version>{fix_version}</version>",
            "fix_type": "maven_version_bump",
        })

    if not fixes_applied:
        return []

    out_dir = output_dir or repo_path
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pom.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pom_content)

    manifest_relative_path = os.path.relpath(java_manifest_path, repo_path)
    for fix in fixes_applied:
        fix["fixed_path"] = out_path
        fix["requirements_relative_path"] = manifest_relative_path

    return fixes_applied


# ---------------------------------------------------------------------------
# Dispatcher -- routes each dependency finding to the right ecosystem's
# fixer based on its "ecosystem" field (set by dep_scanner.py: absent
# for Python, "Go" for Go, "Maven" for Java).
# ---------------------------------------------------------------------------
def generate_dependency_fixes(repo_path: str, dep_findings: list, output_dir: str,
                               requirements_file_path: str = None,
                               go_mod_path: str = None, java_manifest_path: str = None) -> list:
    if not dep_findings:
        return []

    python_findings = [f for f in dep_findings if not f.get("ecosystem")]
    go_findings = [f for f in dep_findings if f.get("ecosystem") == "Go"]
    java_findings = [f for f in dep_findings if f.get("ecosystem") == "Maven"]

    fixes = []
    fixes.extend(generate_python_dependency_fixes(repo_path, python_findings, output_dir, requirements_file_path=requirements_file_path))
    fixes.extend(generate_go_dependency_fixes(repo_path, go_findings, go_mod_path=go_mod_path, output_dir=output_dir))
    fixes.extend(generate_java_dependency_fixes(repo_path, java_findings, java_manifest_path=java_manifest_path, output_dir=output_dir))
    return fixes
@observe(name="generate_all_fixes")
def generate_all_fixes(repo_path: str, repo_name: str, enriched_findings: list, output_dir: str = "./proposed_fixes",
                        requirements_file_path: str = None, go_mod_path: str = None, java_manifest_path: str = None,
                        scan_id: str = None) -> dict:
    # session_id=scan_id groups every observation from this scan (every
    # file's groq-code-fix / bedrock-critic generation, and every
    # critic_score) under one filterable unit in Langfuse -- this is
    # what actually answers "show me the Langfuse trace for scan X"
    # instead of matching by timestamp + repo name by eye. Safe to call
    # unconditionally (a no-op when Langfuse isn't configured, per the
    # SDK's own docs: "No exceptions are raised"), and must wrap the
    # ENTIRE function body -- propagate_attributes only affects spans
    # created after it's entered, so wrapping just the first few lines
    # would silently exclude every file processed afterward.
    with propagate_attributes(
        session_id=scan_id,
        tags=[f"scan:{scan_id}"] if scan_id else None,
        metadata={"repo_name": repo_name},
    ):
        return _generate_all_fixes_impl(repo_path, repo_name, enriched_findings, output_dir, requirements_file_path, go_mod_path, java_manifest_path, scan_id)


def _generate_all_fixes_impl(repo_path: str, repo_name: str, enriched_findings: list, output_dir: str,
                              requirements_file_path: str, go_mod_path: str, java_manifest_path: str, scan_id: str) -> dict:
    code_findings = [f for f in enriched_findings if f.get("finding_type") == "code"]
    dep_findings  = [f for f in enriched_findings if f.get("finding_type") == "dependency"]

    repo_output_dir = os.path.join(output_dir, repo_name)
    os.makedirs(repo_output_dir, exist_ok=True)

    code_fixes = generate_code_fixes(repo_path, code_findings, repo_output_dir, scan_id=scan_id)
    dependency_fixes = generate_dependency_fixes(
        repo_path, dep_findings, repo_output_dir,
        requirements_file_path=requirements_file_path,
        go_mod_path=go_mod_path, java_manifest_path=java_manifest_path,
    )

    summary_lines = [f"# Proposed Fixes: {repo_name}\n"]
    summary_lines.append(f"## Code Fixes ({len(code_fixes)} file(s))\n")
    for fix in code_fixes:
        summary_lines.append(f"### `{fix['relative_path']}`")
        summary_lines.append(f"- Findings addressed: {fix['findings_addressed']}")
        summary_lines.append(f"- Explanation: {fix['explanation']}")
        if fix.get("hf_revision_facts"):
            for model, sha in fix["hf_revision_facts"].items():
                summary_lines.append(f"- Verified HF revision used: `{model}` -> `{sha}`")

        verification = fix.get("verification")
        if verification:
            if not verification.get("syntax_valid"):
                summary_lines.append(f"- ❌ VERIFICATION FAILED: fixed code is not valid Python")
            elif verification.get("bandit_still_flags"):
                summary_lines.append(f"- ❌ VERIFICATION FAILED: bandit still detects rule(s) {verification.get('remaining_issues')} after the fix")
            else:
                summary_lines.append(f"- ✅ Verified: syntax valid, bandit no longer flags the original issue(s)")

            pyflakes_info = verification.get("pyflakes")
            if pyflakes_info and pyflakes_info.get("has_critical"):
                summary_lines.append(f"- ❌ VERIFICATION FAILED: pyflakes found a real bug (undefined name or similar)")
            elif pyflakes_info is not None:
                summary_lines.append(f"- ✅ pyflakes: no correctness-breaking issues found")

            coverage_info = verification.get("rule_coverage")
            if coverage_info and coverage_info.get("applicable", True):
                if coverage_info.get("ok"):
                    summary_lines.append(f"- ✅ Rule coverage: critic confirmed every original finding rule was addressed")
                else:
                    summary_lines.append(f"- ⚠️ Rule coverage INCOMPLETE: critic did not confirm rule(s) {coverage_info.get('missing')} were resolved -- verdict downgraded")

            import_check = verification.get("import_check")
            if import_check:
                if import_check.get("skipped"):
                    summary_lines.append(f"- ⚪ Isolated import check SKIPPED: {import_check.get('error')}")
                elif import_check.get("import_ok"):
                    summary_lines.append(f"- ✅ Isolated import check: fixed file imports/runs without error")
                else:
                    summary_lines.append(f"- ❌ VERIFICATION FAILED: fixed file does not import/run -- {import_check.get('error')}")

            existing_tests = verification.get("existing_tests")
            if existing_tests:
                summary_text = str(existing_tests.get("summary", ""))
                was_skipped = summary_text.startswith("skipped")
                if was_skipped:
                    summary_lines.append(f"- ⚪ Existing test check SKIPPED (candidate already failed an earlier check): {summary_text}")
                elif existing_tests.get("tests_found") and existing_tests.get("tests_passed") is True:
                    summary_lines.append(f"- ✅ Existing test suite: passes with this fix applied")
                elif existing_tests.get("tests_found") and existing_tests.get("tests_passed") is False:
                    summary_lines.append(f"- ❌ VERIFICATION FAILED: existing test suite fails with this fix applied -- {summary_text}")
                elif not existing_tests.get("tests_found"):
                    summary_lines.append(f"- ⚪ Existing test check ran -- no tests found in repo to verify against")

        if fix.get("warnings"):
            for w in fix["warnings"]:
                summary_lines.append(f"- ⚠️ WARNING: {w}")
        critique = fix.get("critique")
        if critique and critique.get("verdict") != "unavailable":
            summary_lines.append(f"- 🔍 Critic score: {critique.get('score')}/10 ({critique.get('verdict')})")
            for p in critique.get("problems", []):
                if isinstance(p, dict):
                    tag = "🔴 CRITICAL" if p.get("severity") == "critical" else ("🟡 moderate" if p.get("severity") == "moderate" else "⚪ minor")
                    summary_lines.append(f"  - {tag}: {p.get('text', '')}")
                else:
                    summary_lines.append(f"  - Problem noted: {p}")
        summary_lines.append(f"- Proposed file: `{fix['fixed_path']}`\n")

    summary_lines.append(f"## Dependency Fixes ({len(dependency_fixes)} package(s))\n")
    for fix in dependency_fixes:
        tag = "DIRECT" if fix.get("fix_type") == "direct" else "TRANSITIVE (new pin)"
        summary_lines.append(f"- [{tag}] `{fix['package']}`: `{fix['old_line']}` -> `{fix['new_line']}`")

    summary_path = os.path.join(repo_output_dir, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return {
        "code_fixes": code_fixes,
        "dependency_fixes": dependency_fixes,
        "output_dir": repo_output_dir,
        "summary_path": summary_path,
    }