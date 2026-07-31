"""
tests/unit/test_dep_scanner.py
---------------------------------
Tests run_dependency_scan() and _find_requirements_file() against mocked
subprocess output and a real temp-directory tree -- never invokes real
pip-audit, so this suite works in CI without it installed.
"""
import json
import os
import subprocess
from unittest.mock import patch, MagicMock

from tools import dep_scanner


def _fake_pip_audit_result(stdout_dict, returncode=0):
    result = MagicMock()
    result.stdout = json.dumps(stdout_dict)
    result.returncode = returncode
    return result


class TestFindRequirementsFile:
    def test_finds_file_at_repo_root(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
        found = dep_scanner._find_requirements_file(str(tmp_path))
        assert found == str(tmp_path / "requirements.txt")

    def test_finds_file_in_common_subdir(self, tmp_path):
        # Regression case named directly in the file's own docstring:
        # we45/Vulnerable-Flask-App keeps requirements.txt under app/.
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "requirements.txt").write_text("flask==1.0\n")
        found = dep_scanner._find_requirements_file(str(tmp_path))
        assert found == str(app_dir / "requirements.txt")

    def test_root_takes_priority_over_subdir(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("root\n")
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "requirements.txt").write_text("nested\n")
        found = dep_scanner._find_requirements_file(str(tmp_path))
        assert found == str(tmp_path / "requirements.txt")

    def test_returns_none_when_absent(self, tmp_path):
        assert dep_scanner._find_requirements_file(str(tmp_path)) is None

    def test_requirements_in_also_matches(self, tmp_path):
        (tmp_path / "requirements.in").write_text("requests\n")
        found = dep_scanner._find_requirements_file(str(tmp_path))
        assert found == str(tmp_path / "requirements.in")


class TestRunDependencyScan:
    def test_no_requirements_file_is_not_an_error(self, tmp_path):
        result = dep_scanner.run_dependency_scan(str(tmp_path))
        assert result["success"] is True
        assert result["requirements_found"] is False
        assert result["findings"] == []

    def test_calls_pip_audit_with_dash_r_flag(self, tmp_path):
        # Regression test for the exact bug documented in the file's own
        # header: an earlier version used `--path` (audits an installed
        # env) instead of `-r <file>` (audits a requirements file's
        # declared deps), silently producing false "0 vulnerable" results.
        (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
        with patch("subprocess.run", return_value=_fake_pip_audit_result({"dependencies": []})) as mock_run:
            dep_scanner.run_dependency_scan(str(tmp_path))
        called_cmd = mock_run.call_args[0][0]
        assert "-r" in called_cmd
        assert "--path" not in called_cmd
        req_file_index = called_cmd.index("-r") + 1
        assert called_cmd[req_file_index] == str(tmp_path / "requirements.txt")

    def test_parses_vulnerable_dependency(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
        audit_json = {
            "dependencies": [
                {
                    "name": "requests", "version": "2.20.0",
                    "vulns": [
                        {"id": "CVE-2023-1234", "description": "Cert validation bug",
                         "fix_versions": ["2.31.0"]}
                    ],
                }
            ]
        }
        with patch("subprocess.run", return_value=_fake_pip_audit_result(audit_json)):
            result = dep_scanner.run_dependency_scan(str(tmp_path))

        assert result["success"] is True
        assert result["total"] == 1
        finding = result["findings"][0]
        assert finding["package"] == "requests"
        assert finding["installed_version"] == "2.20.0"
        assert finding["fix_version"] == "2.31.0"
        assert finding["vulns"][0]["id"] == "CVE-2023-1234"

    def test_dependency_with_no_vulns_excluded(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("safe-pkg==1.0\n")
        audit_json = {"dependencies": [{"name": "safe-pkg", "version": "1.0", "vulns": []}]}
        with patch("subprocess.run", return_value=_fake_pip_audit_result(audit_json)):
            result = dep_scanner.run_dependency_scan(str(tmp_path))
        assert result["total"] == 0

    def test_first_available_fix_version_used_when_multiple_vulns(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("pkg==1.0\n")
        audit_json = {
            "dependencies": [{
                "name": "pkg", "version": "1.0",
                "vulns": [
                    {"id": "CVE-1", "description": "d1", "fix_versions": []},
                    {"id": "CVE-2", "description": "d2", "fix_versions": ["1.5"]},
                ],
            }]
        }
        with patch("subprocess.run", return_value=_fake_pip_audit_result(audit_json)):
            result = dep_scanner.run_dependency_scan(str(tmp_path))
        assert result["findings"][0]["fix_version"] == "1.5"
        assert len(result["findings"][0]["vulns"]) == 2

    def test_pip_audit_not_installed(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = dep_scanner.run_dependency_scan(str(tmp_path))
        assert result["success"] is False
        assert "not installed" in result["error"]

    def test_timeout(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pip-audit", timeout=240)):
            result = dep_scanner.run_dependency_scan(str(tmp_path))
        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_empty_stdout_returns_success_no_findings(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
        empty_result = MagicMock()
        empty_result.stdout = ""
        with patch("subprocess.run", return_value=empty_result):
            result = dep_scanner.run_dependency_scan(str(tmp_path))
        assert result["success"] is True
        assert result["total"] == 0

    def test_malformed_json(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.20.0\n")
        bad_result = MagicMock()
        bad_result.stdout = "{not valid"
        with patch("subprocess.run", return_value=bad_result):
            result = dep_scanner.run_dependency_scan(str(tmp_path))
        assert result["success"] is False
        assert "parse pip-audit" in result["error"]