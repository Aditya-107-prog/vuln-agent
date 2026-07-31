"""
tests/integration/test_org_ui.py
---------------------------------
Flask test-client integration tests for the HTML organizations pages
(/orgs, /orgs/<id>) and the org-awareness added to the existing pages
(scan form org selector, repos-page scope column).

These pages are thin wrappers around the already-tested JSON API in
test_organizations.py -- these tests check what actually renders in
the HTML (visibility of controls by role, 404s for non-members, etc.),
not the underlying permission logic, which is covered there.

Mirrors the same real-HTTP-through-the-test-client approach used by
test_organizations.py and test_web_app.py -- no mocks, real (disposable,
per-test) Postgres database.
"""
import pytest

from models import db, User, Organization, Membership, Repo


def _get_json(resp):
    return resp.get_json()


def _create_org(client, name="Acme Corp"):
    resp = client.post("/organizations", json={"name": name})
    return _get_json(resp)["id"]


def _login_as(client, email, password="testpass123"):
    client.post("/login", data={"email": email, "password": password}, follow_redirects=True)


class TestOrgsListPage:
    def test_unauthenticated_redirects_to_login(self, client):
        resp = client.get("/orgs", follow_redirects=False)
        assert resp.status_code in (302, 401)

    def test_no_orgs_shows_empty_state(self, client, signup):
        signup(client, "alice@test.com")
        resp = client.get("/orgs")
        assert resp.status_code == 200
        assert b"aren&#39;t a member" in resp.data or b"aren't a member" in resp.data

    def test_lists_org_name_role_and_member_count(self, client, signup):
        signup(client, "alice@test.com")
        _create_org(client, "Acme Corp")
        resp = client.get("/orgs")
        assert b"Acme Corp" in resp.data
        assert b"owner" in resp.data
        # one member (alice) so far
        assert b">1<" in resp.data

    def test_does_not_list_orgs_you_do_not_belong_to(self, client, signup):
        signup(client, "alice@test.com")
        _create_org(client, "Alice Org")
        client.get("/logout")

        signup(client, "bob@test.com")
        resp = client.get("/orgs")
        assert b"Alice Org" not in resp.data

    def test_manage_link_points_at_org_detail_page(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client, "Acme Corp")
        resp = client.get("/orgs")
        assert f'/orgs/{org_id}'.encode() in resp.data


class TestOrgDetailPage:
    def test_non_member_gets_404(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.get("/logout")

        signup(client, "outsider@test.com")
        resp = client.get(f"/orgs/{org_id}")
        assert resp.status_code == 404

    def test_unauthenticated_redirects_to_login(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.get("/logout")

        resp = client.get(f"/orgs/{org_id}", follow_redirects=False)
        assert resp.status_code in (302, 401)

    def test_member_list_shows_all_emails(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")

        _login_as(client, "alice@test.com")
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})

        resp = client.get(f"/orgs/{org_id}")
        assert b"alice@test.com" in resp.data
        assert b"bob@test.com" in resp.data

    def test_owner_sees_invite_form_and_role_dropdowns(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)

        resp = client.get(f"/orgs/{org_id}")
        assert b'id="invite-email"' in resp.data
        assert b'id="invite-role"' in resp.data
        # owner can grant admin, so the invite role selector should offer it
        assert b'<option value="admin">admin</option>' in resp.data
        # owner-level role changes render as a <select>, not plain text
        assert b'onchange="changeRole(' in resp.data

    def test_owner_can_invite_as_admin_option_present(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        resp = client.get(f"/orgs/{org_id}")
        # role select for inviting should contain both member and admin
        assert b'<option value="member">member</option>' in resp.data
        assert b'<option value="admin">admin</option>' in resp.data

    def test_plain_member_does_not_see_invite_form_or_role_controls(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")

        _login_as(client, "alice@test.com")
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        client.get("/logout")

        _login_as(client, "bob@test.com")
        resp = client.get(f"/orgs/{org_id}")
        assert resp.status_code == 200
        assert b'id="invite-email"' not in resp.data
        assert b'onchange="changeRole(' not in resp.data
        assert b'onclick="removeMember(' not in resp.data

    def test_admin_sees_invite_form_without_admin_role_option(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")

        _login_as(client, "alice@test.com")
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "admin"})
        client.get("/logout")

        _login_as(client, "bob@test.com")
        resp = client.get(f"/orgs/{org_id}")
        assert b'id="invite-email"' in resp.data
        # admin (not owner) can only invite as member -- no admin option offered
        assert b'<option value="admin">admin</option>' not in resp.data
        # admin does not get owner-level role-change dropdowns
        assert b'onchange="changeRole(' not in resp.data

    def test_admin_can_remove_plain_member_but_not_owner(self, client, signup, app):
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
            alice_id = User.query.filter_by(email="alice@test.com").first().id
            carol_id = User.query.filter_by(email="carol@test.com").first().id
        client.get("/logout")

        _login_as(client, "bob@test.com")
        resp = client.get(f"/orgs/{org_id}")
        # bob (admin) should see a remove control for carol (member)...
        assert f"removeMember({org_id}, {carol_id}".encode() in resp.data
        # ...but not for alice (owner), since bob isn't an owner himself
        assert f"removeMember({org_id}, {alice_id}".encode() not in resp.data

    def test_shows_your_role(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client)
        resp = client.get(f"/orgs/{org_id}")
        assert b"your role: owner" in resp.data


class TestScanFormOrgSelector:
    def test_no_orgs_shows_only_personal_option(self, client, signup):
        signup(client, "alice@test.com")
        resp = client.get("/")
        assert b'id="org-select"' in resp.data
        assert b"Personal (only visible to you)" in resp.data

    def test_org_membership_adds_option_to_selector(self, client, signup):
        signup(client, "alice@test.com")
        org_id = _create_org(client, "Acme Corp")
        resp = client.get("/")
        assert f'<option value="{org_id}">Acme Corp</option>'.encode() in resp.data

    def test_selector_only_shows_your_own_orgs(self, client, signup):
        signup(client, "alice@test.com")
        _create_org(client, "Alice Org")
        client.get("/logout")

        signup(client, "bob@test.com")
        resp = client.get("/")
        assert b"Alice Org" not in resp.data


class TestReposPageScopeColumn:
    def test_personal_repo_shows_personal_scope(self, client, signup, app):
        signup(client, "alice@test.com")
        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            repo = Repo(user_id=alice.id, target="https://github.com/x/y", name="solo-repo")
            db.session.add(repo)
            db.session.commit()

        resp = client.get("/repos")
        assert b"solo-repo" in resp.data
        assert b"Personal" in resp.data

    def test_org_repo_shows_org_name_as_scope(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _create_org(client, "Acme Corp")
        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            repo = Repo(user_id=alice.id, target="https://github.com/x/y",
                        name="team-repo", organization_id=org_id)
            db.session.add(repo)
            db.session.commit()

        resp = client.get("/repos")
        assert b"team-repo" in resp.data
        assert b"Acme Corp" in resp.data
