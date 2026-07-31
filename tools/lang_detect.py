"""
tools/lang_detect.py
---------------------
Detects which language(s) a target repo contains, so code_scanner.py
and fix_generator.py know which scanner/fixer path to use. A repo can
be multi-language (e.g. a Python backend with a Go microservice) --
detect_languages() returns ALL languages found, not just one, since
findings/fixes are already per-file and per-language elsewhere in the
pipeline (each finding carries its own "language" tag going forward).

Detection is marker-file based (go.mod, pom.xml/build.gradle,
requirements.txt/setup.py/pyproject.toml) rather than purely
extension-based, because a stray .java or .go file vendored/copied
into an otherwise-Python repo shouldn't trigger a full second-language
scan pass -- a real marker file means the repo actually treats that
language as a first-class part of the project.

Falls back to extension-count heuristics ONLY if no marker files are
found at all (e.g. a bare folder of loose .go files with no go.mod --
still worth scanning, just less confidently "this is a Go project").
"""

import os

from logging_config import get_logger

logger = get_logger(__name__)

# Directories never worth walking into for marker files -- same
# exclusion list code_scanner.py already uses for bandit's --exclude,
# kept in sync here so language detection doesn't get confused by
# vendored dependency trees (e.g. a Go module's vendor/ directory
# often contains its own go.mod-less .go files, and node_modules can
# contain arbitrary language fixtures in test folders).
EXCLUDED_DIRS = {".venv", "venv", "env", ".env", "node_modules", "dist", "build", ".git", "vendor", "target"}

# language -> marker filenames that confidently indicate that language
# is a first-class part of this repo (not just an incidental file).
LANGUAGE_MARKERS = {
    "go": ["go.mod"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "python": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
}

# Fallback: language -> file extensions, used only when NO marker file
# for ANY language is found anywhere in the repo.
LANGUAGE_EXTENSIONS = {
    "go": (".go",),
    "java": (".java",),
    "python": (".py",),
}


def _find_marker_languages(repo_path: str) -> set:
    found = set()
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        files_set = set(files)
        for lang, markers in LANGUAGE_MARKERS.items():
            if lang in found:
                continue
            if any(m in files_set for m in markers):
                found.add(lang)
        if len(found) == len(LANGUAGE_MARKERS):
            break  # every known language already confirmed, no need to keep walking
    return found


def _find_extension_languages(repo_path: str, min_files: int = 1) -> set:
    """Fallback path: count files per extension. min_files=1 is
    deliberately permissive -- a single loose .go file with no go.mod
    is still worth scanning, just via the weaker signal. Only used
    when marker-based detection found nothing at all, so being
    permissive here doesn't risk drowning a real marker-based result
    in noise from e.g. one stray test fixture.
    """
    counts = {lang: 0 for lang in LANGUAGE_EXTENSIONS}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for f in files:
            for lang, exts in LANGUAGE_EXTENSIONS.items():
                if f.endswith(exts):
                    counts[lang] += 1
    return {lang for lang, c in counts.items() if c >= min_files}


def detect_languages(repo_path: str) -> list:
    """Returns a sorted list of languages detected in the repo, e.g.
    ["go", "python"]. Never returns an empty list for a repo that
    contains any recognized source files -- falls back to extension
    counting if no marker files exist. Returns [] only for a genuinely
    empty/unrecognized repo (e.g. docs-only)."""
    if not os.path.isdir(repo_path):
        logger.error(f"[lang_detect] {repo_path} is not a directory")
        return []

    marker_langs = _find_marker_languages(repo_path)
    if marker_langs:
        logger.info(f"[lang_detect] Detected via marker files: {sorted(marker_langs)}")
        return sorted(marker_langs)

    ext_langs = _find_extension_languages(repo_path)
    if ext_langs:
        logger.info(f"[lang_detect] No marker files found -- detected via file extensions (weaker signal): {sorted(ext_langs)}")
        return sorted(ext_langs)

    logger.info(f"[lang_detect] No recognized language markers or source files found in {repo_path}")
    return []
