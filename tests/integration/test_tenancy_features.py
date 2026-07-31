"""
tests/integration/test_tenancy_features.py
--------------------------------------------
Covers the three tenancy extensions added on top of the base org model:
  1. Repo.restricted + RepoAccess grants (fine-grained repo visibility)
  2. can_push grants gating the PR-approval checkpoint specifically
  3. OrgInvite (pending invites for emails with no account yet)
  4. Organization.default_scan_frequency

Same real-HTTP-through-the-test-client approach as the rest of the
integration suite.
"""
import web_app as web_app_module
from models import db, User, Organization, Membership, Repo, RepoAccess, OrgInvite


def _get_json(resp):
    return resp.get_json()


def _login_as(client, email, password="testpass123"):
    client.post("/login", data={"email": email, "password": password}, follow_redirects=True)


def _create_org(client, name="Acme Corp"):
    resp = client.post("/organizations", json={"name": name})
    return _get_json(resp)["id"]


class TestPendingInvites:
    def test_inviting_unknown_email_creates_pending_invite(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)

        resp = client.post(f"/organizations/{org_id}/members", json={"email": "ghost@test.com", "role": "member"})
        assert resp.status_code == 201
        data = _get_json(resp)
        assert data["pending"] is True

    def test_duplicate_pending_invite_rejected(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.post(f"/organizations/{org_id}/members", json={"email": "ghost@test.com", "role": "member"})

        resp = client.post(f"/organizations/{org_id}/members", json={"email": "ghost@test.com", "role": "member"})
        assert resp.status_code == 409

    def test_signup_consumes_matching_pending_invite(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.post(f"/organizations/{org_id}/members", json={"email": "ghost@test.com", "role": "admin"})
        client.get("/logout")

        signup(client, "ghost@test.com")

        resp = client.get("/organizations")
        orgs = _get_json(resp)["organizations"]
        assert any(o["id"] == org_id and o["role"] == "admin" for o in orgs)

        with app.app_context():
            assert OrgInvite.query.filter_by(email="ghost@test.com").count() == 0

    def test_signup_with_no_invites_unaffected(self, client, signup):
        resp = signup(client, "nobody-invited@test.com")
        assert resp.status_code == 200

    def test_list_invites_visible_to_members(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.post(f"/organizations/{org_id}/members", json={"email": "ghost@test.com", "role": "member"})

        resp = client.get(f"/organizations/{org_id}/invites")
        assert resp.status_code == 200
        emails = [i["email"] for i in _get_json(resp)["invites"]]
        assert "ghost@test.com" in emails

    def test_admin_can_revoke_invite(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.post(f"/organizations/{org_id}/members", json={"email": "ghost@test.com", "role": "member"})
        with app.app_context():
            invite_id = OrgInvite.query.filter_by(email="ghost@test.com").first().id

        resp = client.delete(f"/organizations/{org_id}/invites/{invite_id}")
        assert resp.status_code == 200

        resp = client.get(f"/organizations/{org_id}/invites")
        assert _get_json(resp)["invites"] == []

    def test_plain_member_cannot_revoke_invite(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        client.post(f"/organizations/{org_id}/members", json={"email": "ghost@test.com", "role": "member"})
        with app.app_context():
            invite_id = OrgInvite.query.filter_by(email="ghost@test.com").first().id
        client.get("/logout")

        signup(client, "bob@test.com")  # already has account, but signup() is a no-op-ish path here; log in instead
        client.get("/logout")
        _login_as(client, "bob@test.com")

        resp = client.delete(f"/organizations/{org_id}/invites/{invite_id}")
        assert resp.status_code == 403


class TestOrgSettings:
    def test_owner_can_set_default_scan_frequency(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)

        resp = client.post(f"/organizations/{org_id}/settings", json={"default_scan_frequency": "weekly"})
        assert resp.status_code == 200
        assert _get_json(resp)["default_scan_frequency"] == "weekly"

    def test_invalid_frequency_rejected(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        resp = client.post(f"/organizations/{org_id}/settings", json={"default_scan_frequency": "hourly"})
        assert resp.status_code == 400

    def test_plain_member_cannot_change_settings(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")
        _login_as(client, "bob@test.com")

        resp = client.post(f"/organizations/{org_id}/settings", json={"default_scan_frequency": "daily"})
        assert resp.status_code == 403

    def test_new_org_repo_inherits_default_frequency(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.post(f"/organizations/{org_id}/settings", json={"default_scan_frequency": "daily"})

        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            repo = Repo(user_id=alice.id, target="https://github.com/x/y", name="team-repo", organization_id=org_id)
            org = db.session.get(Organization, org_id)
            repo.scan_frequency = org.default_scan_frequency
            db.session.add(repo)
            db.session.commit()
            assert repo.scan_frequency == "daily"


class TestRepoRestrictionAndAccessGrants:
    def _org_with_owner_admin_member(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")
        signup(client, "carol@test.com")
        client.get("/logout")

        _login_as(client, "alice@test.com")
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "admin"})
        client.post(f"/organizations/{org_id}/members", json={"email": "carol@test.com", "role": "member"})

        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            repo = Repo(user_id=alice.id, target="https://github.com/x/y", name="team-repo", organization_id=org_id)
            db.session.add(repo)
            db.session.commit()
            repo_id = repo.id

        return org_id, repo_id

    def test_unrestricted_repo_visible_to_plain_member(self, client, signup, app):
        org_id, repo_id = self._org_with_owner_admin_member(client, signup, app)
        client.get("/logout")
        _login_as(client, "carol@test.com")
        resp = client.get("/repos")
        assert b"team-repo" in resp.data

    def test_restricting_repo_hides_it_from_ungranted_member(self, client, signup, app):
        org_id, repo_id = self._org_with_owner_admin_member(client, signup, app)
        client.post(f"/repos/{repo_id}/access", json={"restricted": True})

        client.get("/logout")
        _login_as(client, "carol@test.com")
        resp = client.get("/repos")
        assert b"team-repo" not in resp.data

    def test_restricted_repo_still_visible_to_admin_and_owner(self, client, signup, app):
        org_id, repo_id = self._org_with_owner_admin_member(client, signup, app)
        client.post(f"/repos/{repo_id}/access", json={"restricted": True})

        resp = client.get("/repos")
        assert b"team-repo" in resp.data  # alice (owner)

        client.get("/logout")
        _login_as(client, "bob@test.com")  # admin
        resp = client.get("/repos")
        assert b"team-repo" in resp.data

    def test_granting_access_restores_visibility(self, client, signup, app):
        org_id, repo_id = self._org_with_owner_admin_member(client, signup, app)
        client.post(f"/repos/{repo_id}/access", json={"restricted": True})
        client.post(f"/repos/{repo_id}/access", json={"email": "carol@test.com", "can_push": False})

        client.get("/logout")
        _login_as(client, "carol@test.com")
        resp = client.get("/repos")
        assert b"team-repo" in resp.data

    def test_revoking_access_hides_it_again(self, client, signup, app):
        org_id, repo_id = self._org_with_owner_admin_member(client, signup, app)
        client.post(f"/repos/{repo_id}/access", json={"restricted": True})
        client.post(f"/repos/{repo_id}/access", json={"email": "carol@test.com", "can_push": False})
        with app.app_context():
            carol_id = User.query.filter_by(email="carol@test.com").first().id

        client.delete(f"/repos/{repo_id}/access/{carol_id}")

        client.get("/logout")
        _login_as(client, "carol@test.com")
        resp = client.get("/repos")
        assert b"team-repo" not in resp.data

    def test_plain_member_cannot_manage_access(self, client, signup, app):
        org_id, repo_id = self._org_with_owner_admin_member(client, signup, app)
        client.get("/logout")
        _login_as(client, "carol@test.com")

        resp = client.post(f"/repos/{repo_id}/access", json={"restricted": True})
        assert resp.status_code == 403

    def test_list_repo_access_shows_restriction_and_grants(self, client, signup, app):
        org_id, repo_id = self._org_with_owner_admin_member(client, signup, app)
        client.post(f"/repos/{repo_id}/access", json={"restricted": True})
        client.post(f"/repos/{repo_id}/access", json={"email": "carol@test.com", "can_push": True})

        resp = client.get(f"/repos/{repo_id}/access/grants")
        data = _get_json(resp)
        assert data["restricted"] is True
        grant = next(g for g in data["grants"] if g["email"] == "carol@test.com")
        assert grant["can_push"] is True


class TestPushPermissionOnPrCheckpoint:
    """can_be_pushed_by / the /approve route's pr_approval gate -- tested
    directly against the model + a hand-constructed SCANS entry rather
    than running the full scan pipeline, since that requires real
    bandit/pip-audit/LLM calls."""

    def _setup_repo_with_member(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.get("/logout")
        signup(client, "carol@test.com")
        client.get("/logout")

        _login_as(client, "alice@test.com")
        client.post(f"/organizations/{org_id}/members", json={"email": "carol@test.com", "role": "member"})

        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            repo = Repo(user_id=alice.id, target="https://github.com/x/y", name="team-repo", organization_id=org_id)
            db.session.add(repo)
            db.session.commit()
            repo_id = repo.id

        return org_id, repo_id

    def test_member_without_can_push_blocked_at_pr_checkpoint(self, client, signup, app):
        org_id, repo_id = self._setup_repo_with_member(client, signup, app)

        client.get("/logout")
        _login_as(client, "carol@test.com")

        scan_id = "test-scan-1"
        web_app_module.SCANS[scan_id] = {
            "status": "waiting_approval", "checkpoint": "pr_approval", "repo_id": repo_id,
            "user_id": None, "events": [], "approval_answer": None,
        }
        try:
            resp = client.post(f"/approve/{scan_id}", json={"answer": "approve"})
            assert resp.status_code == 403
        finally:
            web_app_module.SCANS.pop(scan_id, None)

    def test_member_with_can_push_allowed_at_pr_checkpoint(self, client, signup, app):
        org_id, repo_id = self._setup_repo_with_member(client, signup, app)
        client.post(f"/repos/{repo_id}/access", json={"email": "carol@test.com", "can_push": True})

        client.get("/logout")
        _login_as(client, "carol@test.com")

        scan_id = "test-scan-2"
        web_app_module.SCANS[scan_id] = {
            "status": "waiting_approval", "checkpoint": "pr_approval", "repo_id": repo_id,
            "user_id": None, "events": [], "approval_answer": None,
            "approval_event": __import__("threading").Event(),
        }
        try:
            resp = client.post(f"/approve/{scan_id}", json={"answer": "approve"})
            assert resp.status_code == 200
        finally:
            web_app_module.SCANS.pop(scan_id, None)

    def test_report_checkpoint_unaffected_by_push_rights(self, client, signup, app):
        """The earlier report-approval checkpoint should stay open to
        anyone with repo access, regardless of can_push."""
        org_id, repo_id = self._setup_repo_with_member(client, signup, app)

        client.get("/logout")
        _login_as(client, "carol@test.com")

        scan_id = "test-scan-3"
        web_app_module.SCANS[scan_id] = {
            "status": "waiting_approval", "checkpoint": "report_approval", "repo_id": repo_id,
            "user_id": None, "events": [], "approval_answer": None,
            "approval_event": __import__("threading").Event(),
        }
        try:
            resp = client.post(f"/approve/{scan_id}", json={"answer": "approve"})
            assert resp.status_code == 200
        finally:
            web_app_module.SCANS.pop(scan_id, None)

    def test_reject_at_pr_checkpoint_never_requires_push_rights(self, client, signup, app):
        """Regression test: a member with view-only (no can_push) access
        must be able to REJECT at the PR checkpoint even though they
        can't approve it -- rejecting doesn't open anything, so there's
        nothing to gate. Approving is the only action that needs
        can_push; this used to incorrectly block reject too."""
        org_id, repo_id = self._setup_repo_with_member(client, signup, app)

        client.get("/logout")
        _login_as(client, "carol@test.com")

        scan_id = "test-scan-reject"
        web_app_module.SCANS[scan_id] = {
            "status": "waiting_approval", "checkpoint": "pr_approval", "repo_id": repo_id,
            "user_id": None, "events": [], "approval_answer": None,
            "approval_event": __import__("threading").Event(),
        }
        try:
            resp = client.post(f"/approve/{scan_id}", json={"answer": "reject"})
            assert resp.status_code == 200
        finally:
            web_app_module.SCANS.pop(scan_id, None)

    def test_admin_can_push_without_explicit_grant(self, client, signup, app):
        org_id, repo_id = self._setup_repo_with_member(client, signup, app)
        client.post(f"/organizations/{org_id}/members", json={"email": "carol@test.com", "role": "admin"}) \
            if False else None  # carol is already 'member'; promote instead:
        with app.app_context():
            carol = User.query.filter_by(email="carol@test.com").first()
            m = Membership.query.filter_by(user_id=carol.id, organization_id=org_id).first()
            m.role = "admin"
            db.session.commit()

        client.get("/logout")
        _login_as(client, "carol@test.com")

        scan_id = "test-scan-4"
        web_app_module.SCANS[scan_id] = {
            "status": "waiting_approval", "checkpoint": "pr_approval", "repo_id": repo_id,
            "user_id": None, "events": [], "approval_answer": None,
            "approval_event": __import__("threading").Event(),
        }
        try:
            resp = client.post(f"/approve/{scan_id}", json={"answer": "approve"})
            assert resp.status_code == 200
        finally:
            web_app_module.SCANS.pop(scan_id, None)