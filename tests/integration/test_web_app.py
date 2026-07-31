"""
tests/integration/test_web_app.py
------------------------------------
Flask test-client integration tests. Uses a real (in-memory SQLite) DB
via the `app`/`client` fixtures in conftest.py, but never triggers a
real scan pipeline (no LLM/network calls) -- just auth, tenancy, and
RBAC, which is exactly what we manually verified by hand earlier and
now want locked in as an automated regression test.
"""
import pytest

from models import db, User, Repo, Scan


class TestAuthFlow:
    def test_signup_creates_user_and_logs_in(self, client, signup):
        resp = signup(client, "alice@test.com")
        assert resp.status_code == 200
        with client.session_transaction() as sess:
            assert "_user_id" in sess

    def test_first_signup_becomes_admin(self, client, signup, app):
        signup(client, "first@test.com")
        with app.app_context():
            user = User.query.filter_by(email="first@test.com").first()
            assert user.role == "admin"

    def test_second_signup_is_member(self, client, signup, app):
        signup(client, "first@test.com")
        client.get("/logout")
        signup(client, "second@test.com")
        with app.app_context():
            second = User.query.filter_by(email="second@test.com").first()
            assert second.role == "member"

    def test_duplicate_email_rejected(self, client, signup):
        signup(client, "dup@test.com")
        client.get("/logout")
        resp = signup(client, "dup@test.com")
        assert b"already exists" in resp.data

    def test_login_with_wrong_password_rejected(self, client, signup, login):
        signup(client, "alice@test.com", password="correct-pw")
        client.get("/logout")
        resp = login(client, "alice@test.com", password="wrong-pw")
        assert b"Invalid email or password" in resp.data

    def test_logout_clears_session(self, client, signup):
        signup(client, "alice@test.com")
        client.get("/logout")
        with client.session_transaction() as sess:
            assert "_user_id" not in sess

    def test_root_requires_login(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (302, 401)


class TestTenancyIsolation:
    """Regression coverage for the exact class of bug (IDOR) the manual
    tenancy audit checked for by hand -- one user must never be able to
    view, modify, or delete another user's repos/scans."""

    def _make_repo_for(self, app, email, repo_name="victim-repo"):
        with app.app_context():
            user = User.query.filter_by(email=email).first()
            repo = Repo(user_id=user.id, target="https://github.com/x/y", name=repo_name)
            db.session.add(repo)
            db.session.commit()
            return repo.id

    def test_user_cannot_delete_another_users_repo(self, client, signup, app):
        signup(client, "victim@test.com")
        client.get("/logout")
        repo_id = self._make_repo_for(app, "victim@test.com")

        signup(client, "attacker@test.com")
        resp = client.post(f"/repos/{repo_id}/delete")
        assert resp.status_code == 404

        with app.app_context():
            assert db.session.get(Repo, repo_id) is not None  # still there

    def test_user_cannot_rescan_another_users_repo(self, client, signup, app):
        signup(client, "victim@test.com")
        client.get("/logout")
        repo_id = self._make_repo_for(app, "victim@test.com")

        signup(client, "attacker@test.com")
        resp = client.post(f"/repos/{repo_id}/scan")
        assert resp.status_code == 404

    def test_history_only_shows_own_scans(self, client, signup, app):
        signup(client, "victim@test.com")
        client.get("/logout")
        repo_id = self._make_repo_for(app, "victim@test.com")
        with app.app_context():
            user = User.query.filter_by(email="victim@test.com").first()
            scan = Scan(id="victim-scan-1", user_id=user.id, repo_id=repo_id, status="done")
            db.session.add(scan)
            db.session.commit()

        signup(client, "attacker@test.com")
        resp = client.get("/history")
        assert b"victim-repo" not in resp.data

        client.get("/logout")
        login_resp = client.post("/login", data={"email": "victim@test.com", "password": "testpass123"},
                                  follow_redirects=True)
        resp = client.get("/history")
        assert b"victim-repo" in resp.data

    def test_report_not_served_for_other_users_scan(self, client, signup, app):
        signup(client, "victim@test.com")
        client.get("/logout")
        repo_id = self._make_repo_for(app, "victim@test.com")
        with app.app_context():
            user = User.query.filter_by(email="victim@test.com").first()
            scan = Scan(id="victim-scan-2", user_id=user.id, repo_id=repo_id,
                        status="done", report_path="/tmp/does-not-matter.html")
            db.session.add(scan)
            db.session.commit()

        signup(client, "attacker@test.com")
        resp = client.get("/report/victim-scan-2")
        assert resp.status_code == 404


class TestRBACAdminRoute:
    def test_member_gets_403_on_admin_scans(self, client, signup):
        signup(client, "admin-account@test.com")  # first signup -> admin
        client.get("/logout")
        signup(client, "member-account@test.com")  # second signup -> member
        resp = client.get("/admin/scans")
        assert resp.status_code == 403

    def test_admin_gets_200_on_admin_scans(self, client, signup):
        signup(client, "admin-account@test.com")  # first signup -> admin
        resp = client.get("/admin/scans")
        assert resp.status_code == 200
        assert resp.get_json()["total"] == 0

    def test_admin_scans_shows_data_from_all_users(self, client, signup, app):
        signup(client, "admin-account@test.com")
        client.get("/logout")
        signup(client, "member-account@test.com")

        with app.app_context():
            member = User.query.filter_by(email="member-account@test.com").first()
            repo = Repo(user_id=member.id, target="https://github.com/x/y", name="member-repo")
            db.session.add(repo)
            db.session.commit()
            scan = Scan(id="member-scan-1", user_id=member.id, repo_id=repo.id, status="done")
            db.session.add(scan)
            db.session.commit()

        client.get("/logout")
        client.post("/login", data={"email": "admin-account@test.com", "password": "testpass123"},
                    follow_redirects=True)
        resp = client.get("/admin/scans")
        data = resp.get_json()
        assert data["total"] == 1
        assert data["scans"][0]["user_email"] == "member-account@test.com"

    def test_unauthenticated_user_cannot_reach_admin_route(self, client):
        resp = client.get("/admin/scans", follow_redirects=False)
        assert resp.status_code in (302, 401)
