"""
tests/integration/test_organizations.py
------------------------------------------
Flask test-client integration tests for the organization/membership
routes added in this round: create org, list orgs, list/add/remove
members, change roles -- plus the org-based repo visibility this all
exists to support (a team repo is visible to every member, a personal
repo stays visible only to its creator).

Mirrors the same tenancy-testing approach used in test_web_app.py:
real HTTP requests through the Flask test client, against a real
(disposable, per-test) Postgres database -- not mocks.
"""
import pytest

from models import db, User, Organization, Membership, Repo


def _get_json(resp):
    return resp.get_json()


class TestCreateAndListOrganizations:
    def test_create_org_makes_creator_owner(self, client, signup):
        signup(client, "alice@test.com")
        resp = client.post("/organizations", json={"name": "Acme Corp"})
        assert resp.status_code == 201
        data = _get_json(resp)
        assert data["name"] == "Acme Corp"
        assert data["role"] == "owner"

    def test_create_org_requires_name(self, client, signup):
        signup(client, "alice@test.com")
        resp = client.post("/organizations", json={"name": ""})
        assert resp.status_code == 400

    def test_list_organizations_shows_only_your_own(self, client, signup):
        signup(client, "alice@test.com")
        client.post("/organizations", json={"name": "Alice Org"})
        client.get("/logout")

        signup(client, "bob@test.com")
        client.post("/organizations", json={"name": "Bob Org"})

        resp = client.get("/organizations")
        names = [o["name"] for o in _get_json(resp)["organizations"]]
        assert names == ["Bob Org"]

    def test_user_with_no_orgs_gets_empty_list(self, client, signup):
        signup(client, "alice@test.com")
        resp = client.get("/organizations")
        assert _get_json(resp)["organizations"] == []

    def test_unauthenticated_cannot_create_org(self, client):
        resp = client.post("/organizations", json={"name": "x"}, follow_redirects=False)
        assert resp.status_code in (302, 401)


class TestOrgMembership:
    def _create_org(self, client, name="Acme Corp"):
        resp = client.post("/organizations", json={"name": name})
        return _get_json(resp)["id"]

    def test_owner_can_add_member(self, client, signup):
        signup(client, "alice@test.com")
        org_id = self._create_org(client)
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")
        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)

        resp = client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        assert resp.status_code == 201
        assert _get_json(resp)["role"] == "member"

    def test_plain_member_cannot_add_others(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = self._create_org(client)
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")

        # alice adds bob as a plain member
        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        client.get("/logout")

        # bob (member) tries to add a third person -- should be forbidden
        client.post("/login", data={"email": "bob@test.com", "password": "testpass123"}, follow_redirects=True)
        signup(client, "carol@test.com")  # just to have a real user to add
        client.get("/logout")
        client.post("/login", data={"email": "bob@test.com", "password": "testpass123"}, follow_redirects=True)

        resp = client.post(f"/organizations/{org_id}/members", json={"email": "carol@test.com", "role": "member"})
        assert resp.status_code == 403

    def test_admin_cannot_add_someone_as_admin_only_owner_can(self, client, signup):
        signup(client, "alice@test.com")
        org_id = self._create_org(client)
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")
        signup(client, "carol@test.com")
        client.get("/logout")

        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "admin"})
        client.get("/logout")

        # bob is admin, tries to add carol AS ADMIN -- only an owner can do that
        client.post("/login", data={"email": "bob@test.com", "password": "testpass123"}, follow_redirects=True)
        resp = client.post(f"/organizations/{org_id}/members", json={"email": "carol@test.com", "role": "admin"})
        assert resp.status_code == 403

        # but bob CAN add carol as a plain member
        resp = client.post(f"/organizations/{org_id}/members", json={"email": "carol@test.com", "role": "member"})
        assert resp.status_code == 201

    def test_non_member_gets_404_not_403_listing_members(self, client, signup):
        """404, not 403 -- a non-member shouldn't even learn the org exists."""
        signup(client, "alice@test.com")
        org_id = self._create_org(client)
        client.get("/logout")

        signup(client, "outsider@test.com")
        resp = client.get(f"/organizations/{org_id}/members")
        assert resp.status_code == 404

    def test_duplicate_member_add_rejected(self, client, signup):
        signup(client, "alice@test.com")
        org_id = self._create_org(client)
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")

        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        resp = client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        assert resp.status_code == 409


class TestRoleChangesAndOwnerProtection:
    def _create_org_with_owner_and_member(self, client, signup):
        signup(client, "alice@test.com")
        resp = client.post("/organizations", json={"name": "Acme"})
        org_id = _get_json(resp)["id"]
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")
        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        return org_id

    def test_owner_can_promote_member_to_admin(self, client, signup, app):
        org_id = self._create_org_with_owner_and_member(client, signup)
        with app.app_context():
            bob = User.query.filter_by(email="bob@test.com").first()
            bob_id = bob.id
        resp = client.post(f"/organizations/{org_id}/members/{bob_id}/role", json={"role": "admin"})
        assert resp.status_code == 200
        assert _get_json(resp)["role"] == "admin"

    def test_member_cannot_change_anyones_role(self, client, signup, app):
        org_id = self._create_org_with_owner_and_member(client, signup)
        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            alice_id = alice.id
        client.get("/logout")
        client.post("/login", data={"email": "bob@test.com", "password": "testpass123"}, follow_redirects=True)

        resp = client.post(f"/organizations/{org_id}/members/{alice_id}/role", json={"role": "member"})
        assert resp.status_code == 403

    def test_cannot_demote_last_owner(self, client, signup, app):
        org_id = self._create_org_with_owner_and_member(client, signup)
        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            alice_id = alice.id
        # alice (the only owner) tries to demote herself
        resp = client.post(f"/organizations/{org_id}/members/{alice_id}/role", json={"role": "member"})
        assert resp.status_code == 409

    def test_can_demote_owner_if_another_owner_exists(self, client, signup, app):
        org_id = self._create_org_with_owner_and_member(client, signup)
        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            bob = User.query.filter_by(email="bob@test.com").first()
            alice_id, bob_id = alice.id, bob.id

        # promote bob to owner too, then demoting alice should be fine now
        client.post(f"/organizations/{org_id}/members/{bob_id}/role", json={"role": "owner"})
        resp = client.post(f"/organizations/{org_id}/members/{alice_id}/role", json={"role": "member"})
        assert resp.status_code == 200


class TestRemoveMember:
    def test_admin_can_remove_a_plain_member(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _get_json(client.post("/organizations", json={"name": "Acme"}))["id"]
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")
        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})

        with app.app_context():
            bob_id = User.query.filter_by(email="bob@test.com").first().id

        resp = client.delete(f"/organizations/{org_id}/members/{bob_id}")
        assert resp.status_code == 200

        resp = client.get(f"/organizations/{org_id}/members")
        emails = [m["email"] for m in _get_json(resp)["members"]]
        assert "bob@test.com" not in emails

    def test_admin_cannot_remove_an_owner(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _get_json(client.post("/organizations", json={"name": "Acme"}))["id"]
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")
        signup(client, "carol@test.com")
        client.get("/logout")

        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "admin"})
        client.post(f"/organizations/{org_id}/members", json={"email": "carol@test.com", "role": "member"})
        with app.app_context():
            alice_id = User.query.filter_by(email="alice@test.com").first().id
        client.get("/logout")

        # bob (admin) tries to remove alice (owner) -- forbidden
        client.post("/login", data={"email": "bob@test.com", "password": "testpass123"}, follow_redirects=True)
        resp = client.delete(f"/organizations/{org_id}/members/{alice_id}")
        assert resp.status_code == 403

    def test_cannot_remove_last_owner(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _get_json(client.post("/organizations", json={"name": "Acme"}))["id"]
        with app.app_context():
            alice_id = User.query.filter_by(email="alice@test.com").first().id

        resp = client.delete(f"/organizations/{org_id}/members/{alice_id}")
        assert resp.status_code == 409


class TestOrgRepoTenancy:
    """The actual point of all this: a repo attached to an organization
    should be visible to every member, and a personal repo should stay
    invisible to everyone else -- tested through real HTTP requests this
    time (test_web_app.py's TestTenancyIsolation covers the DB-level
    equivalent for purely personal repos)."""

    def test_org_member_can_see_teammates_org_repo_in_history(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _get_json(client.post("/organizations", json={"name": "Acme"}))["id"]
        client.get("/logout")
        signup(client, "bob@test.com")
        client.get("/logout")

        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})

        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            repo = Repo(user_id=alice.id, target="https://github.com/x/y",
                        name="team-repo", organization_id=org_id)
            db.session.add(repo)
            db.session.commit()

        client.get("/logout")
        client.post("/login", data={"email": "bob@test.com", "password": "testpass123"}, follow_redirects=True)
        resp = client.get("/repos")
        assert b"team-repo" in resp.data

    def test_non_member_cannot_see_org_repo(self, client, signup, app):
        signup(client, "alice@test.com")
        org_id = _get_json(client.post("/organizations", json={"name": "Acme"}))["id"]
        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            repo = Repo(user_id=alice.id, target="https://github.com/x/y",
                        name="team-repo", organization_id=org_id)
            db.session.add(repo)
            db.session.commit()
        client.get("/logout")

        signup(client, "outsider@test.com")
        resp = client.get("/repos")
        assert b"team-repo" not in resp.data

    def test_personal_repo_stays_invisible_to_org_mates(self, client, signup, app):
        """An org existing at all must not leak a member's PERSONAL
        (non-org) repos to their org-mates."""
        signup(client, "alice@test.com")
        org_id = _get_json(client.post("/organizations", json={"name": "Acme"}))["id"]
        with app.app_context():
            alice = User.query.filter_by(email="alice@test.com").first()
            personal_repo = Repo(user_id=alice.id, target="https://github.com/x/personal", name="alice-personal")
            db.session.add(personal_repo)
            db.session.commit()
        client.get("/logout")

        signup(client, "bob@test.com")
        client.get("/logout")
        client.post("/login", data={"email": "alice@test.com", "password": "testpass123"}, follow_redirects=True)
        client.post(f"/organizations/{org_id}/members", json={"email": "bob@test.com", "role": "member"})
        client.get("/logout")

        client.post("/login", data={"email": "bob@test.com", "password": "testpass123"}, follow_redirects=True)
        resp = client.get("/repos")
        assert b"alice-personal" not in resp.data
