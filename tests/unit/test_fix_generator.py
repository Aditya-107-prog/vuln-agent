"""
tests/unit/test_fix_generator.py
------------------------------------
Covers the verification logic in fix_generator.py that decides whether
a proposed code fix actually works -- specifically the baseline-aware
existing-test-suite check (_try_run_tests / _run_existing_tests_with_baseline).

Two kinds of tests here, deliberately:

1. Mocked-subprocess tests (TestTryRunTestsMocked, TestBaselineComparison)
   -- fast, no real pytest subprocess spawned, test the DECISION LOGIC:
   given a certain combination of baseline/fixed outcomes, is the right
   verdict produced? This is what most of this file is.

2. Real-fixture tests (TestRealPytestExecution) -- no mocking, actually
   writes a tiny real repo to disk with a real pytest test file and lets
   _try_run_tests / _run_existing_tests_with_baseline really invoke
   `python -m pytest` as a subprocess against it. These are slower and
   need pytest installed (which it will be, since that's what's running
   this suite) but they're the only thing that proves the *real*
   subprocess/pytest mechanics work, not just the branching logic around
   a mocked result.

Nothing here needs a Groq/Langfuse API key or network access -- the
functions under test here never call the LLM or the critic.
"""
import os
import subprocess
import textwrap
from unittest.mock import patch, MagicMock

import pytest

from tools import fix_generator as fg


def _mock_run(returncode, stdout="", stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


# ---------------------------------------------------------------------------
# _try_run_tests: decision logic, subprocess mocked
# ---------------------------------------------------------------------------
class TestTryRunTestsMocked:
    def test_no_tests_in_repo(self):
        with patch.object(fg, "_make_repo_copy_with_fix", return_value="/fake/tmp"), \
             patch("subprocess.run", return_value=_mock_run(5)), \
             patch("shutil.rmtree"):
            result = fg._try_run_tests("/fake/repo", "app.py", "print('hi')")
        assert result["tests_found"] is False
        assert result["tests_passed"] is None
        assert "no tests found" in result["summary"]

    def test_tests_pass(self):
        collect_ok = _mock_run(0)
        run_ok = _mock_run(0, stdout="5 passed in 0.10s")
        with patch.object(fg, "_make_repo_copy_with_fix", return_value="/fake/tmp"), \
             patch("subprocess.run", side_effect=[collect_ok, run_ok]), \
             patch("shutil.rmtree"):
            result = fg._try_run_tests("/fake/repo", "app.py", "print('hi')")
        assert result["tests_found"] is True
        assert result["tests_passed"] is True

    def test_tests_fail(self):
        collect_ok = _mock_run(0)
        run_fail = _mock_run(1, stdout="1 failed, 4 passed in 0.12s")
        with patch.object(fg, "_make_repo_copy_with_fix", return_value="/fake/tmp"), \
             patch("subprocess.run", side_effect=[collect_ok, run_fail]), \
             patch("shutil.rmtree"):
            result = fg._try_run_tests("/fake/repo", "app.py", "print('hi')")
        assert result["tests_found"] is True
        assert result["tests_passed"] is False

    def test_collection_error_flagged_distinctly(self):
        # Exit code 2 = collection interrupted, e.g. ImportError in a test
        # file. This must be distinguishable from a real test failure so
        # baseline comparison can tell "environment gap" apart from
        # "the fix broke something."
        collect_error = _mock_run(2, stdout="ImportError: No module named 'flask_sqlalchemy'\n")
        with patch.object(fg, "_make_repo_copy_with_fix", return_value="/fake/tmp"), \
             patch("subprocess.run", return_value=collect_error), \
             patch("shutil.rmtree"):
            result = fg._try_run_tests("/fake/repo", "app.py", "print('hi')")
        assert result["tests_found"] is False
        assert result.get("collection_error") is True

    def test_copy_failure_is_skipped_not_a_failure(self):
        with patch.object(fg, "_make_repo_copy_with_fix", return_value=None):
            result = fg._try_run_tests("/fake/repo", "app.py", "print('hi')")
        assert result["tests_found"] is False
        assert result["tests_passed"] is None
        assert "skipped" in result["summary"]

    def test_timeout_is_inconclusive_not_a_failure(self):
        with patch.object(fg, "_make_repo_copy_with_fix", return_value="/fake/tmp"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=60)), \
             patch("shutil.rmtree"):
            result = fg._try_run_tests("/fake/repo", "app.py", "print('hi')")
        assert result["tests_passed"] is None
        assert "timed out" in result["summary"]

    def test_pytest_not_installed(self):
        with patch.object(fg, "_make_repo_copy_with_fix", return_value="/fake/tmp"), \
             patch("subprocess.run", side_effect=FileNotFoundError), \
             patch("shutil.rmtree"):
            result = fg._try_run_tests("/fake/repo", "app.py", "print('hi')")
        assert result["tests_found"] is False
        assert "not installed" in result["summary"]

    def test_temp_dir_always_cleaned_up(self):
        with patch.object(fg, "_make_repo_copy_with_fix", return_value="/fake/tmp"), \
             patch("subprocess.run", return_value=_mock_run(5)), \
             patch("shutil.rmtree") as mock_rmtree:
            fg._try_run_tests("/fake/repo", "app.py", "print('hi')")
        mock_rmtree.assert_called_once_with("/fake/tmp", ignore_errors=True)


# ---------------------------------------------------------------------------
# _run_existing_tests_with_baseline: the actual "don't blame the fix for
# a pre-existing environment gap" comparison logic
# ---------------------------------------------------------------------------
class TestBaselineComparison:
    def _patch_try_run_tests(self, baseline_result, fixed_result):
        # baseline is called first (original code), fixed second.
        return patch.object(fg, "_try_run_tests", side_effect=[baseline_result, fixed_result])

    def test_baseline_and_fixed_both_pass(self):
        passing = {"tests_found": True, "tests_passed": True, "summary": "5 passed"}
        with self._patch_try_run_tests(passing, passing):
            result = fg._run_existing_tests_with_baseline("/repo", "app.py", "old code", "new code")
        assert result["tests_passed"] is True

    def test_baseline_passes_fixed_fails_is_a_real_regression(self):
        baseline = {"tests_found": True, "tests_passed": True, "summary": "5 passed"}
        fixed = {"tests_found": True, "tests_passed": False, "summary": "1 failed, 4 passed"}
        with self._patch_try_run_tests(baseline, fixed):
            result = fg._run_existing_tests_with_baseline("/repo", "app.py", "old code", "new code")
        assert result["tests_passed"] is False
        assert "skipped" not in result["summary"]

    def test_both_fail_same_collection_error_is_environment_gap_not_blamed(self):
        # This is the exact real bug documented in fix_generator.py's own
        # comments: pytest collection failed on BOTH the original and
        # fixed file (missing dependency in our sandbox) -- must not be
        # reported as the fix's fault.
        same_error = {"tests_found": False, "tests_passed": None,
                      "summary": "test collection errored: ModuleNotFoundError",
                      "collection_error": True}
        with self._patch_try_run_tests(same_error, same_error):
            result = fg._run_existing_tests_with_baseline("/repo", "app.py", "old code", "new code")
        assert "skipped" in result["summary"]
        assert "not attributable to the fix" in result["summary"]

    def test_fixed_fails_but_baseline_collection_error_is_still_skipped(self):
        # Different shape: fixed run produces a real test failure (not a
        # collection error), but baseline couldn't even collect tests at
        # all -- still shouldn't be blamed on the fix, since we have no
        # working baseline to compare against.
        baseline = {"tests_found": False, "tests_passed": None,
                    "summary": "test collection errored: ImportError",
                    "collection_error": True}
        fixed = {"tests_found": True, "tests_passed": False, "summary": "1 failed"}
        with self._patch_try_run_tests(baseline, fixed):
            result = fg._run_existing_tests_with_baseline("/repo", "app.py", "old code", "new code")
        assert result["tests_passed"] is None
        assert "skipped" in result["summary"]

    def test_no_tests_found_at_all(self):
        no_tests = {"tests_found": False, "tests_passed": None, "summary": "no tests found in repo"}
        with self._patch_try_run_tests(no_tests, no_tests):
            result = fg._run_existing_tests_with_baseline("/repo", "app.py", "old code", "new code")
        assert result["tests_found"] is False

    def test_baseline_always_run_against_original_content_first(self):
        """Regression check on call order -- if baseline and fixed were
        ever swapped, a real regression would be misread as an
        environment gap (or vice versa)."""
        calls = []

        def fake_try_run_tests(repo_path, relative_path, code):
            calls.append(code)
            return {"tests_found": True, "tests_passed": True, "summary": "ok"}

        with patch.object(fg, "_try_run_tests", side_effect=fake_try_run_tests):
            fg._run_existing_tests_with_baseline("/repo", "app.py", "ORIGINAL_CODE", "FIXED_CODE")

        assert calls == ["ORIGINAL_CODE", "FIXED_CODE"]


# ---------------------------------------------------------------------------
# Real (non-mocked) pytest execution against an actual fixture repo on disk
# ---------------------------------------------------------------------------
class TestRealPytestExecution:
    """No mocking here -- these actually spawn `python -m pytest` as a
    real subprocess against a real temp copy of a fixture repo. Slower
    than the mocked tests above, but they're what actually proves the
    subprocess wiring (cwd, exit codes, timeouts) works, not just the
    decision logic around a fake result.
    """

    def _write_fixture_repo(self, tmp_path, app_code, test_code):
        (tmp_path / "app.py").write_text(app_code)
        (tmp_path / "test_app.py").write_text(test_code)
        return str(tmp_path)

    def test_real_repo_with_passing_tests(self, tmp_path):
        repo = self._write_fixture_repo(
            tmp_path,
            app_code="def add(a, b):\n    return a + b\n",
            test_code=textwrap.dedent("""
                from app import add
                def test_add():
                    assert add(2, 3) == 5
            """),
        )
        result = fg._try_run_tests(repo, "app.py", "def add(a, b):\n    return a + b\n")
        assert result["tests_found"] is True
        assert result["tests_passed"] is True

    def test_real_repo_with_a_fix_that_breaks_a_test(self, tmp_path):
        original_code = "def add(a, b):\n    return a + b\n"
        broken_fix = "def add(a, b):\n    return a - b\n"  # deliberately wrong
        repo = self._write_fixture_repo(
            tmp_path,
            app_code=original_code,
            test_code=textwrap.dedent("""
                from app import add
                def test_add():
                    assert add(2, 3) == 5
            """),
        )
        result = fg._run_existing_tests_with_baseline(repo, "app.py", original_code, broken_fix)
        assert result["tests_passed"] is False
        assert "skipped" not in result["summary"]

    def test_real_repo_with_no_tests_at_all(self, tmp_path):
        (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n")
        result = fg._try_run_tests(str(tmp_path), "app.py", "def add(a, b):\n    return a + b\n")
        assert result["tests_found"] is False
        assert result["tests_passed"] is None

    def test_real_repo_where_test_file_cant_import_a_missing_module(self, tmp_path):
        # Simulates the documented real-world case: the target repo's own
        # test file imports something not installed in our sandbox
        # (e.g. flask_sqlalchemy). This must surface as a collection
        # error, not a false "tests failed" verdict -- and baseline
        # comparison should then skip it entirely since it's identical
        # for both the original and fixed code.
        original_code = "def add(a, b):\n    return a + b\n"
        fixed_code = "def add(a, b):\n    return a + b  # comment added\n"
        repo = self._write_fixture_repo(
            tmp_path,
            app_code=original_code,
            test_code=textwrap.dedent("""
                import this_module_does_not_exist_anywhere
                from app import add
                def test_add():
                    assert add(2, 3) == 5
            """),
        )
        result = fg._run_existing_tests_with_baseline(repo, "app.py", original_code, fixed_code)
        assert "skipped" in result["summary"]
        assert "not attributable to the fix" in result["summary"]
