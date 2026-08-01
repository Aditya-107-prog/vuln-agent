"""
scan_requirements.py
---------------------
Scans your project's own .py files (skipping venv, tests, and the
proposed_fixes/pr_workdir scan-output folders) for top-level imports,
maps them to actual pip package names, and cross-references against
your full `pip freeze` dump to produce a trimmed requirements.txt
containing only what the app actually needs -- with the exact pinned
versions currently installed.

Usage (from your project root, venv activated):
    python scan_requirements.py

It reads full_freeze.txt (paste your full `pip freeze` output there)
and writes requirements.min.txt. Review that file, then rename it to
requirements.txt once you're happy with it.
"""

import ast
import os
import re
import sys

# Folders to skip entirely -- not part of the deployed app.
SKIP_DIRS = {
    "venv", ".venv", "env", "__pycache__", ".git",
    "tests", "test",
    "proposed_fixes", "pr_workdir", "node_modules",
}

# import-name -> actual-pypi-package-name, for the common mismatches.
# Anything not listed here is assumed to match its import name.
KNOWN_MAP = {
    "github": "PyGithub",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "flask_sqlalchemy": "Flask-SQLAlchemy",
    "flask_login": "Flask-Login",
    "flask_migrate": "Flask-Migrate",
    "flask_cors": "flask-cors",
    "sqlalchemy": "SQLAlchemy",
    "yaml": "PyYAML",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
    "docx": "python-docx",
    "google": None,   # handled specially below (google.* namespace pkgs)
    "jose": "python-jose",
    "OpenSSL": "pyOpenSSL",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "engineio": "python-engineio",
    "socketio": "python-socketio",
    "multipart": "python-multipart",
    "win32api": "pywin32",
    "win32com": "pywin32",
}

# Python stdlib modules never need to go in requirements.txt.
STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()


def find_py_files(root="."):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def extract_imports(filepath):
    names = set()
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read(), filename=filepath)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # skip relative imports
                names.add(node.module.split(".")[0])
    return names


def main():
    root = "."
    all_imports = set()
    for filepath in find_py_files(root):
        all_imports |= extract_imports(filepath)

    all_imports -= STDLIB
    # drop obviously-local single-word module names that match files/folders
    # in this project itself (best-effort; harmless if it over-keeps a few).
    local_modules = {
        os.path.splitext(f)[0]
        for f in os.listdir(root)
        if f.endswith(".py")
    }
    all_imports -= local_modules

    resolved = set()
    for name in sorted(all_imports):
        if name == "google":
            resolved.add("google")  # flag for manual review, see note below
            continue
        pkg = KNOWN_MAP.get(name, name)
        if pkg:
            resolved.add(pkg)

    if not os.path.exists("full_freeze.txt"):
        print("Missing full_freeze.txt -- paste your `pip freeze` output into")
        print("a file called full_freeze.txt in this folder, then re-run.")
        sys.exit(1)

    with open("full_freeze.txt", "r", encoding="utf-8") as fh:
        freeze_lines = [l.strip() for l in fh if l.strip()]

    freeze_map = {}
    for line in freeze_lines:
        m = re.match(r"^([A-Za-z0-9_.\-]+)==", line)
        if m:
            freeze_map[m.group(1).lower()] = line

    matched, unmatched = [], []
    for pkg in sorted(resolved, key=str.lower):
        key = pkg.lower()
        if key in freeze_map:
            matched.append(freeze_map[key])
        elif pkg == "google":
            unmatched.append("# 'google' namespace import found -- check which")
            unmatched.append("# google-* package(s) you actually need (e.g. google-generativeai)")
        else:
            unmatched.append(f"# UNRESOLVED (add manually, check exact pip name): {pkg}")

    with open("requirements.min.txt", "w", encoding="utf-8") as out:
        out.write("# Auto-generated from actual project imports.\n")
        out.write("# Review before using -- see any UNRESOLVED lines below.\n\n")
        out.write("gunicorn\n")  # not imported directly, but required to run in prod
        out.write("\n".join(matched))
        out.write("\n")
        if unmatched:
            out.write("\n# --- needs manual review ---\n")
            out.write("\n".join(unmatched))
            out.write("\n")

    print(f"Scanned {len(list(find_py_files(root)))} .py files.")
    print(f"Found {len(matched)} matched packages, {len(unmatched)} needing manual review.")
    print("Wrote requirements.min.txt -- review it, then rename to requirements.txt.")


if __name__ == "__main__":
    main()