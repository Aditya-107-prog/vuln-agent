"""
Tool 2: Local Directory Reader
--------------------------------
Validates a local folder path, walks the file tree,
and returns a summary of what's inside — focusing on Python files.
"""

import os

from logging_config import get_logger

logger = get_logger(__name__)


# File extensions we care about for this project
LANGUAGE_MAP = {
    ".py": "Python",
    ".txt": "Text",
    ".md": "Markdown",
    ".cfg": "Config",
    ".toml": "Config",
    ".ini": "Config",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".json": "JSON",
    ".sh": "Shell",
    ".env": "Env",
}


def read_local_directory(path: str) -> dict:
    """
    Validate a local directory and summarise its contents.

    Args:
        path: Absolute or relative path to a local folder

    Returns:
        {
            "success": True,
            "path": "/abs/path/to/folder",
            "file_count": 42,
            "python_files": ["src/main.py", "src/utils.py", ...],
            "language_summary": {"Python": 10, "YAML": 3, ...},
            "has_requirements": True,   # found requirements.txt?
            "error": None
        }
    """

    # --- Validate the path exists and is a directory ---
    path = os.path.abspath(path)  # convert to absolute path

    if not os.path.exists(path):
        return {
            "success": False,
            "path": path,
            "file_count": 0,
            "python_files": [],
            "language_summary": {},
            "has_requirements": False,
            "error": f"Path does not exist: {path}"
        }

    if not os.path.isdir(path):
        return {
            "success": False,
            "path": path,
            "file_count": 0,
            "python_files": [],
            "language_summary": {},
            "has_requirements": False,
            "error": f"Path is not a directory: {path}"
        }

    logger.info(f"Scanning {path}")

    # --- Walk the file tree ---
    file_count = 0
    python_files = []
    language_summary = {}
    has_requirements = False

    # Folders to skip — not useful for scanning
    SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}

    for root, dirs, files in os.walk(path):
        # Modify dirs in-place to skip unwanted folders
        # (this tells os.walk not to descend into them)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            file_count += 1
            ext = os.path.splitext(filename)[1].lower()
            lang = LANGUAGE_MAP.get(ext, "Other")

            # Count by language
            language_summary[lang] = language_summary.get(lang, 0) + 1

            # Track Python files specifically (relative path for readability)
            if ext == ".py":
                rel_path = os.path.relpath(os.path.join(root, filename), path)
                python_files.append(rel_path)

            # Check for requirements file
            if filename in ("requirements.txt", "requirements.in", "Pipfile", "pyproject.toml"):
                has_requirements = True

    logger.info(f"Found {file_count} files, {len(python_files)} Python files")

    return {
        "success": True,
        "path": path,
        "file_count": file_count,
        "python_files": python_files,
        "language_summary": language_summary,
        "has_requirements": has_requirements,
        "error": None
    }


# --- Quick sanity check ---
if __name__ == "__main__":
    import sys

    # Test on current directory if no arg given
    target = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"=== Test: scanning '{target}' ===")
    result = read_local_directory(target)

    if result["success"]:
        print(f"\nPath:         {result['path']}")
        print(f"Total files:  {result['file_count']}")
        print(f"Requirements: {result['has_requirements']}")
        print(f"\nLanguage breakdown:")
        for lang, count in sorted(result["language_summary"].items(), key=lambda x: -x[1]):
            print(f"  {lang:<12} {count} files")
        print(f"\nPython files ({len(result['python_files'])}):")
        for f in result["python_files"][:10]:  # show first 10
            print(f"  {f}")
        if len(result["python_files"]) > 10:
            print(f"  ... and {len(result['python_files']) - 10} more")
    else:
        print(f"Error: {result['error']}")