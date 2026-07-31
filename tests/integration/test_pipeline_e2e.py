"""
tests/integration/test_pipeline_e2e.py
------------------------------------------
Runs the REAL agent.graph pipeline, real bandit, and a real interrupt/
resume checkpoint against a tiny fixture repo with a known, genuine
vulnerability -- as opposed to every other test file in this suite,
which tests one module/function at a time.

This is the test that would have caught an integration bug like a
field-name mismatch between code_scanner.py and report_generator.py
automatically, since it exercises the real node-to-node wiring instead
of a mocked stand-in of it.

What's REAL here (not mocked):
- agent.graph.build_graph() and its actual node wiring/routing
- The router's local-path detection
- A real `bandit` subprocess run against a real fixture file with a
  genuine hardcoded-password vulnerability
- LangGraph's interrupt()/Command(resume=...) checkpoint mechanics
- report_generator.py's real deterministic table assembly

What's mocked, and why:
- run_dependency_scan -- avoids a real pip-audit subprocess + a real
  network call to the OSV vulnerability database. Unit-tested separately
  in test_dep_scanner.py.
- run_cve_enrichment -- avoids a real network call. We deliberately make
  it raise, which exercises nodes.py's OWN documented fallback path
  (raw findings get a default severity/empty osv_ids) rather than
  reimplementing enrichment logic in the test.
- report_generator._call_groq -- avoids a real, costly LLM API call.
  Unit-tested with more thorough coverage in test_report_generator.py.
- generate_all_fixes -- avoids real LLM-based fix generation + critique.
  Deliberately returns zero fixes here so the pipeline naturally reaches
  its own real "no fixes were generated -- nothing to PR" branch in
  pr_review_node, needing no second interrupt and no GitHub calls at all.

A local (non-GitHub) target is used specifically because pr_review_node
already auto-skips the PR step for local targets (state["input_type"] !=
"github") -- so this test never touches tools.github_pr, keeping the
scope to exactly one real interrupt/resume checkpoint.
"""
import os
import uuid
from unittest.mock import patch, MagicMock

import pytest
from langgraph.types import Command

from agent.graph import build_graph


VULNERABLE_APP_CODE = '''
import subprocess

DB_PASSWORD = "hunter2_hardcoded"  # bandit: B105 hardcoded password

def run_admin_command(cmd):
    # bandit: B602 subprocess call with shell=True
    return subprocess.call(cmd, shell=True)
'''


def _mock_groq_response(text):
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def vulnerable_fixture_repo(tmp_path):
    (tmp_path / "app.py").write_text(VULNERABLE_APP_CODE)
    return str(tmp_path)


@pytest.fixture
def mocked_external_calls():
    """Patches exactly the external/expensive boundaries described in
    this file's module docstring -- everything else in the pipeline
    runs for real."""
    llm_output = (
        "===EXEC_SUMMARY===\n"
        "This repository has findings that should be addressed before deployment.\n"
        "===RECOMMENDATIONS===\n"
        "- Remove the hardcoded credential\n- Avoid shell=True in subprocess calls\n"
        "===END==="
    )
    with patch("agent.nodes.run_dependency_scan",
               return_value={"success": True, "findings": [], "total": 0,
                             "requirements_found": False, "requirements_file": None, "error": None}), \
         patch("agent.nodes.run_cve_enrichment", side_effect=Exception("network disabled in test")), \
         patch("tools.report_generator._call_groq", return_value=_mock_groq_response(llm_output)), \
         patch("agent.nodes.generate_all_fixes",
               return_value={"code_fixes": [], "dependency_fixes": [], "output_dir": "./proposed_fixes"}):
        yield


class TestPipelineEndToEnd:
    def _run_config(self):
        return {"configurable": {"thread_id": str(uuid.uuid4())}}

    def test_full_run_report_approved(self, vulnerable_fixture_repo, mocked_external_calls, tmp_path):
        os.environ["AGENT_OUTPUT_DIR"] = str(tmp_path / "reports")
        graph = build_graph()
        config = self._run_config()

        result = graph.invoke({"target": vulnerable_fixture_repo}, config=config)

        # Should have paused at the human_review checkpoint, not run to completion yet.
        assert "__interrupt__" in result
        interrupt_payload = result["__interrupt__"][0].value
        assert interrupt_payload["checkpoint"] == "report_approval"
        assert interrupt_payload["total_findings"] >= 1  # the real bandit findings

        # Resume, approving the report checkpoint.
        final_state = graph.invoke(Command(resume="approve"), config=config)

        # --- Real bandit actually found the real vulnerabilities ---
        code_findings = final_state.get("code_findings") or []
        assert len(code_findings) >= 1
        found_issue_ids = {f.get("issue") for f in code_findings}
        assert "B105" in found_issue_ids or "B602" in found_issue_ids

        # --- Real report was actually generated to disk ---
        assert final_state.get("report_path")
        assert os.path.exists(final_state["report_path"])

        report_md = final_state.get("report_markdown", "")
        assert "# Security Scan Report" in report_md
        assert "## Findings Breakdown" in report_md
        assert "## Executive Summary" in report_md
        assert "Remove the hardcoded credential" in report_md  # our mocked LLM text landed correctly

        # --- Fix generation ran (mocked to return zero fixes) and the
        # pipeline correctly skipped the PR step without needing a
        # second interrupt or any GitHub calls, since there was nothing
        # to PR. ---
        assert final_state.get("pr_decision") == "reject"
        assert final_state.get("pr_url") is None
        assert final_state.get("error") is None

    def test_human_review_rejection_stops_before_report(self, vulnerable_fixture_repo, mocked_external_calls, tmp_path):
        os.environ["AGENT_OUTPUT_DIR"] = str(tmp_path / "reports")
        graph = build_graph()
        config = self._run_config()

        graph.invoke({"target": vulnerable_fixture_repo}, config=config)
        final_state = graph.invoke(Command(resume="reject"), config=config)

        assert final_state.get("human_decision") == "reject"
        assert final_state.get("report_path") is None
        assert final_state.get("report_markdown") is None

    def test_invalid_target_ends_with_error_no_scan_attempted(self):
        graph = build_graph()
        config = self._run_config()

        final_state = graph.invoke({"target": "/this/path/definitely/does/not/exist"}, config=config)

        assert final_state.get("error") is not None
        assert "Cannot resolve target" in final_state["error"]
        assert final_state.get("code_findings") is None
        assert "__interrupt__" not in final_state  # never reached a checkpoint

    def test_empty_directory_ends_with_error(self, tmp_path, mocked_external_calls):
        graph = build_graph()
        config = self._run_config()

        final_state = graph.invoke({"target": str(tmp_path)}, config=config)

        assert final_state.get("error") is not None
        assert "empty" in final_state["error"].lower()
