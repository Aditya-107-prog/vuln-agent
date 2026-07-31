"""
tests/unit/test_code_scanner.py
---------------------------------
Tests run_code_scan() against mocked subprocess output -- never invokes
real bandit, so this suite works in CI without bandit installed.
"""
import json
import subprocess
from unittest.mock import patch, MagicMock

from tools import code_scanner


def _fake_bandit_result(stdout_dict, returncode=0):
    result = MagicMock()
    result.stdout = json.dumps(stdout_dict)
    result.returncode = returncode
    return result


class TestRunCodeScan:
    def test_parses_findings_into_clean_format(self):
        bandit_json = {
            "results": [
                {
                    "filename": "app.py", "line_number": 12,
                    "issue_severity": "high", "issue_confidence": "medium",
                    "test_id": "B608", "issue_text": "Possible SQL injection",
                    "code": "  cur.execute(query)  \n",
                }
            ]
        }
        with patch("subprocess.run", return_value=_fake_bandit_result(bandit_json)):
            result = code_scanner.run_code_scan("/fake/repo")

        assert result["success"] is True
        assert result["total"] == 1
        finding = result["findings"][0]
        assert finding["file"] == "app.py"
        assert finding["line"] == 12
        assert finding["severity"] == "HIGH"          # upper-cased
        assert finding["confidence"] == "MEDIUM"
        assert finding["issue"] == "B608"              # the field name that
                                                        # report_generator.py
                                                        # must read as rule ID
        assert finding["code_snippet"] == "cur.execute(query)"  # stripped

    def test_no_findings_returns_empty_list(self):
        with patch("subprocess.run", return_value=_fake_bandit_result({"results": []})):
            result = code_scanner.run_code_scan("/fake/repo")
        assert result["success"] is True
        assert result["findings"] == []
        assert result["total"] == 0

    def test_empty_stdout_returns_empty_list_not_error(self):
        empty_result = MagicMock()
        empty_result.stdout = ""
        with patch("subprocess.run", return_value=empty_result):
            result = code_scanner.run_code_scan("/fake/repo")
        assert result["success"] is True
        assert result["total"] == 0
        assert result["error"] is None

    def test_findings_sorted_high_before_medium_before_low(self):
        bandit_json = {
            "results": [
                {"filename": "a.py", "line_number": 1, "issue_severity": "low",
                 "issue_confidence": "high", "test_id": "B1", "issue_text": "t", "code": ""},
                {"filename": "b.py", "line_number": 2, "issue_severity": "high",
                 "issue_confidence": "high", "test_id": "B2", "issue_text": "t", "code": ""},
                {"filename": "c.py", "line_number": 3, "issue_severity": "medium",
                 "issue_confidence": "high", "test_id": "B3", "issue_text": "t", "code": ""},
            ]
        }
        with patch("subprocess.run", return_value=_fake_bandit_result(bandit_json)):
            result = code_scanner.run_code_scan("/fake/repo")
        severities = [f["severity"] for f in result["findings"]]
        assert severities == ["HIGH", "MEDIUM", "LOW"]

    def test_bandit_not_installed(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = code_scanner.run_code_scan("/fake/repo")
        assert result["success"] is False
        assert "not installed" in result["error"]

    def test_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="bandit", timeout=120)):
            result = code_scanner.run_code_scan("/fake/repo")
        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_malformed_json_output(self):
        bad_result = MagicMock()
        bad_result.stdout = "{not valid json"
        with patch("subprocess.run", return_value=bad_result):
            result = code_scanner.run_code_scan("/fake/repo")
        assert result["success"] is False
        assert "parse bandit" in result["error"]

    def test_recursive_flag_excludes_venvs(self):
        with patch("subprocess.run", return_value=_fake_bandit_result({"results": []})) as mock_run:
            code_scanner.run_code_scan("/fake/repo", recursive=True)
        called_cmd = mock_run.call_args[0][0]
        assert "-r" in called_cmd
        assert "--exclude" in called_cmd

    def test_non_recursive_skips_exclude_flag(self):
        with patch("subprocess.run", return_value=_fake_bandit_result({"results": []})) as mock_run:
            code_scanner.run_code_scan("/fake/single_file.py", recursive=False)
        called_cmd = mock_run.call_args[0][0]
        assert "-r" not in called_cmd
        assert "--exclude" not in called_cmd