"""
tests/conftest.py
------------------
Shared fixtures for the whole test suite.

Design notes:
- Unit tests should never need a real DB, a real LLM call, or a real
  bandit/pip-audit/pytest subprocess -- everything external is mocked
  in the unit tests themselves.
- Integration tests (tests/integration/) DO spin up a real Flask app +
  real in-memory SQLite DB via the `app` / `client` fixtures below, but
  never make real network/LLM calls.
"""
import os
import sys
import pytest

# Make the project root importable regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_enriched_findings():
    """A small, deterministic mix of code + dependency findings covering
    every severity, used across report_generator tests."""
    return [
        {
            "finding_type": "code", "file": "app.py", "line": 12,
            "severity": "HIGH", "issue": "B608",
            "description": "Possible SQL injection via string-built query",
            "code_snippet": "cur.execute('SELECT * FROM users WHERE id=' + user_id)",
        },
        {
            "finding_type": "code", "file": "utils.py", "line": 40,
            "severity": "LOW", "issue": "B101",
            "description": "Use of assert detected",
            "code_snippet": "assert user.is_active",
        },
        {
            "finding_type": "dependency", "package": "requests",
            "installed_version": "2.20.0", "fix_version": "2.31.0",
            "severity": "CRITICAL", "osv_ids": ["CVE-2023-1234"],
            "vulns": [{"description": "Improper certificate validation in requests"}],
        },
        {
            "finding_type": "dependency", "package": "pyyaml",
            "installed_version": "5.1", "fix_version": "5.4",
            "severity": "MEDIUM", "osv_ids": [],
            "vulns": [{"description": "Arbitrary code execution via yaml.load"}],
        },
    ]


# ---------------------------------------------------------------------------
# Flask app + DB fixtures (used by integration tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def app():
    """A Flask app instance for integration tests.

    IMPORTANT: now that SQLite has been retired, this runs against
    whatever real Postgres DATABASE_URL points to -- there is no more
    in-memory-SQLite swap for test isolation. To keep tests independent
    of each other (and of any real data you're using that DB for), every
    test gets a clean slate: all tables are dropped and recreated before
    it runs. DO NOT point DATABASE_URL at a database you care about
    while running this test suite -- use a dedicated test database
    (e.g. vuln_agent_test), never your real vuln_agent one.
    """
    os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key")
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL is not set -- integration tests require a real "
                    "Postgres test database (SQLite has been retired). Set "
                    "DATABASE_URL to a DEDICATED test DB before running these.")

    import web_app as web_app_module

    web_app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)

    with web_app_module.app.app_context():
        web_app_module.db.drop_all()
        web_app_module.db.create_all()
        yield web_app_module.app
        web_app_module.db.session.remove()
        web_app_module.db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def _signup(client, email, password="testpass123"):
    return client.post("/signup", data={"email": email, "password": password}, follow_redirects=True)


def _login(client, email, password="testpass123"):
    return client.post("/login", data={"email": email, "password": password}, follow_redirects=True)


@pytest.fixture
def signup():
    return _signup


@pytest.fixture
def login():
    return _login
