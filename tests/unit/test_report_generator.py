"""
tests/unit/test_report_generator.py
------------------------------------
Covers the deterministic (non-LLM) parts of report_generator.py:
- build_findings_breakdown_markdown / build_scan_summary_markdown
- parse_llm_sections (including its fallback path)
- build_fix_results_markdown's heading order and "no fixes attempted" case
- generate_report's assembly, with the LLM call itself mocked out

Nothing here makes a real Groq API call, or needs bandit/pip-audit
installed -- report_generator.py never shells out to either.
"""
import json
from unittest.mock import patch, MagicMock

from tools import report_generator as rg


# ---------------------------------------------------------------------------
# Deterministic table builders
# ---------------------------------------------------------------------------
class TestFindingsBreakdown:
    def test_separates_code_and_dependency_findings(self, sample_enriched_findings):
        md = rg.build_findings_breakdown_markdown(sample_enriched_findings)
        assert "### Code Issues" in md
        assert "### Dependency Vulnerabilities" in md
        assert "app.py" in md and "utils.py" in md
        assert "requests" in md and "pyyaml" in md

    def test_rule_id_comes_from_issue_field(self, sample_enriched_findings):
        # code_scanner.py's actual output key is "issue" (e.g. "B608"),
        # not "rule_id"/"test_id" -- regression test for that exact bug.
        md = rg.build_findings_breakdown_markdown(sample_enriched_findings)
        assert "B608" in md
        assert "B101" in md

    def test_missing_rule_id_falls_back_to_na(self):
        findings = [{"finding_type": "code", "file": "x.py", "line": 1,
                     "severity": "LOW", "description": "no issue field here"}]
        md = rg.build_findings_breakdown_markdown(findings)
        assert "N/A" in md

    def test_empty_code_findings_says_so(self):
        findings = [{"finding_type": "dependency", "package": "foo",
                     "installed_version": "1.0", "fix_version": "1.1",
                     "severity": "LOW", "osv_ids": []}]
        md = rg.build_findings_breakdown_markdown(findings)
        assert "No code issues were found." in md

    def test_empty_dependency_findings_says_so(self):
        findings = [{"finding_type": "code", "file": "x.py", "line": 1,
                     "severity": "LOW", "description": "x", "issue": "B1"}]
        md = rg.build_findings_breakdown_markdown(findings)
        assert "No dependency vulnerabilities were found." in md

    def test_pipe_and_newline_in_description_dont_break_table(self):
        findings = [{"finding_type": "code", "file": "x.py", "line": 1,
                     "severity": "LOW", "issue": "B1",
                     "description": "bad | value\nwith newline"}]
        md = rg.build_findings_breakdown_markdown(findings)
        # Escaped pipe shouldn't create a phantom extra table column, and
        # the newline shouldn't split the row across lines.
        assert "bad \\| value with newline" in md

    def test_osv_ids_joined_or_na(self):
        findings = [
            {"finding_type": "dependency", "package": "a", "installed_version": "1",
             "fix_version": "2", "severity": "HIGH", "osv_ids": ["CVE-1", "CVE-2"]},
            {"finding_type": "dependency", "package": "b", "installed_version": "1",
             "fix_version": "2", "severity": "LOW", "osv_ids": []},
        ]
        md = rg.build_findings_breakdown_markdown(findings)
        assert "CVE-1, CVE-2" in md
        assert "N/A" in md


class TestScanSummary:
    def test_counts_by_severity(self, sample_enriched_findings):
        md = rg.build_scan_summary_markdown(sample_enriched_findings)
        assert "| Critical | 1 |" in md
        assert "| High | 1 |" in md
        assert "| Medium | 1 |" in md
        assert "| Low | 1 |" in md
        assert "| Total | 4 |" in md

    def test_empty_findings_all_zero(self):
        md = rg.build_scan_summary_markdown([])
        assert "| Total | 0 |" in md

    def test_unexpected_severity_label_still_counted(self):
        md = rg.build_scan_summary_markdown([{"severity": "info"}])
        assert "| Info | 1 |" in md
        assert "| Total | 1 |" in md


# ---------------------------------------------------------------------------
# LLM section parsing
# ---------------------------------------------------------------------------
class TestParseLlmSections:
    def test_parses_well_formed_output(self):
        raw = (
            "===EXEC_SUMMARY===\n"
            "Overall posture is moderate. One critical dependency issue.\n"
            "===RECOMMENDATIONS===\n"
            "- Upgrade requests\n- Review SQL query construction\n"
            "===END==="
        )
        result = rg.parse_llm_sections(raw)
        assert "moderate" in result["exec_summary"]
        assert "Upgrade requests" in result["recommendations"]

    def test_falls_back_gracefully_on_malformed_output(self):
        raw = "The model just wrote free-form prose with no markers at all."
        result = rg.parse_llm_sections(raw)
        assert result["exec_summary"] == raw.strip()
        assert result["recommendations"] == ""

    def test_missing_recommendations_marker_falls_back(self):
        raw = "===EXEC_SUMMARY===\nSome summary text\n(no recommendations marker)"
        result = rg.parse_llm_sections(raw)
        # Falls back to dumping everything in exec_summary rather than crashing.
        assert result["recommendations"] == ""
        assert "Some summary text" in result["exec_summary"]


# ---------------------------------------------------------------------------
# Fix results markdown structure
# ---------------------------------------------------------------------------
class TestBuildFixResultsMarkdown:
    def test_no_fixes_attempted(self):
        md = rg.build_fix_results_markdown("repo", [], [], [])
        assert "No automated fixes were attempted" in md
        # Both headings should still appear even with nothing to report.
        assert "## Verification Methodology" in md
        assert "## Fix Results" in md

    def test_verification_methodology_precedes_fix_results(self):
        code_fixes = [{"relative_path": "app.py", "findings_addressed": 1,
                       "critic_score": 9, "critic_verdict": "pass"}]
        md = rg.build_fix_results_markdown("repo", code_fixes, [], [])
        assert md.index("## Verification Methodology") < md.index("## Fix Results")


# ---------------------------------------------------------------------------
# generate_report assembly (LLM call mocked)
# ---------------------------------------------------------------------------
class TestGenerateReportAssembly:
    def _mock_groq_response(self, text):
        mock_choice = MagicMock()
        mock_choice.message.content = text
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def test_report_structure_and_marker_present(self, sample_enriched_findings, tmp_path):
        llm_output = (
            "===EXEC_SUMMARY===\nRepo has one critical dependency issue.\n"
            "===RECOMMENDATIONS===\n- Upgrade requests immediately\n===END==="
        )
        with patch.object(rg, "_call_groq", return_value=self._mock_groq_response(llm_output)):
            result = rg.generate_report("test-repo", sample_enriched_findings, output_dir=str(tmp_path))

        assert result["success"] is True
        md = result["report_markdown"]

        # Heading order: Header -> Exec Summary -> Findings Breakdown ->
        # [fix marker] -> Recommendations -> Scan Summary.
        assert md.index("# Security Scan Report: test-repo") == 0
        assert md.index("## Executive Summary") < md.index("## Findings Breakdown")
        assert md.index("## Findings Breakdown") < md.index(rg.FIX_RESULTS_MARKER)
        assert md.index(rg.FIX_RESULTS_MARKER) < md.index("## Recommendations")
        assert md.index("## Recommendations") < md.index("## Scan Summary")

        assert "Upgrade requests immediately" in md
        assert "Repo has one critical dependency issue" in md

    def test_groq_error_returns_failure_not_exception(self, sample_enriched_findings, tmp_path):
        with patch.object(rg, "_call_groq", side_effect=Exception("API is down")):
            result = rg.generate_report("test-repo", sample_enriched_findings, output_dir=str(tmp_path))
        assert result["success"] is False
        assert "API is down" in result["error"]


# ---------------------------------------------------------------------------
# append_fix_results_to_report: marker splice vs fallback append
# ---------------------------------------------------------------------------
class TestAppendFixResults:
    def test_splices_into_marker_not_appended_at_end(self, tmp_path):
        report_path = tmp_path / "report.html"
        report_path.write_text("<html>placeholder</html>")

        report_markdown = (
            "# Security Scan Report: repo\n\n"
            "## Executive Summary\n\nSummary text.\n\n"
            "## Findings Breakdown\n\nSome findings.\n\n"
            f"{rg.FIX_RESULTS_MARKER}\n\n"
            "## Recommendations\n\nDo things.\n\n"
            "## Scan Summary\n\nCounts here.\n"
        )

        result = rg.append_fix_results_to_report(
            report_path=str(report_path),
            report_markdown=report_markdown,
            repo_name="repo",
            code_fixes=[{"relative_path": "app.py", "findings_addressed": 1,
                         "critic_score": 9, "critic_verdict": "pass"}],
            withheld_fixes=[],
            dependency_fixes=[],
        )

        assert result["success"] is True
        updated = result["report_markdown"]
        assert rg.FIX_RESULTS_MARKER not in updated
        # Fix Results section should land BETWEEN Findings Breakdown and
        # Recommendations, not tacked on after Scan Summary.
        assert updated.index("## Findings Breakdown") < updated.index("## Fix Results")
        assert updated.index("## Fix Results") < updated.index("## Recommendations")

    def test_missing_marker_falls_back_to_append(self, tmp_path):
        """Defensive path for reports generated before the marker existed."""
        report_path = tmp_path / "report.html"
        report_path.write_text("<html>placeholder</html>")

        report_markdown = "# Security Scan Report: repo\n\nNo marker in this old report.\n"

        result = rg.append_fix_results_to_report(
            report_path=str(report_path),
            report_markdown=report_markdown,
            repo_name="repo",
            code_fixes=[], withheld_fixes=[], dependency_fixes=[],
        )
        assert result["success"] is True
        assert "## Fix Results" in result["report_markdown"]

    def test_missing_report_path_fails_gracefully(self):
        result = rg.append_fix_results_to_report(
            report_path="/nonexistent/path/report.html",
            report_markdown="whatever",
            repo_name="repo",
            code_fixes=[], withheld_fixes=[], dependency_fixes=[],
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()