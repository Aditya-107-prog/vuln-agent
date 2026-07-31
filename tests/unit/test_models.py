"""
tests/unit/test_models.py
---------------------------
Tests the SQLAlchemy models directly against an in-memory SQLite DB --
no Flask app/routes involved (that's covered in
tests/integration/test_web_app.py). Covers password hashing, the RBAC
role column/default, and the User -> Repo -> Scan relationships/cascades.
"""
import pytest
from flask import Flask

from models import db, User, Repo, Scan


@pytest.fixture
def db_session():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    db.init_app(app)
    with app.app_context():
        db.create_all()
        yield db.session
        db.session.remove()
        db.drop_all()


class TestUserPasswordHashing:
    def test_password_is_hashed_not_stored_plaintext(self, db_session):
        user = User(email="a@test.com")
        user.set_password("hunter2")
        assert user.password_hash != "hunter2"

    def test_check_password_correct(self, db_session):
        user = User(email="a@test.com")
        user.set_password("hunter2")
        assert user.check_password("hunter2") is True

    def test_check_password_incorrect(self, db_session):
        user = User(email="a@test.com")
        user.set_password("hunter2")
        assert user.check_password("wrong-password") is False


class TestUserRoleDefault:
    def test_default_role_is_member(self, db_session):
        user = User(email="a@test.com")
        user.set_password("x")
        db_session.add(user)
        db_session.commit()
        assert user.role == "member"
        assert user.is_admin() is False

    def test_explicit_admin_role(self, db_session):
        user = User(email="a@test.com", role="admin")
        user.set_password("x")
        db_session.add(user)
        db_session.commit()
        assert user.is_admin() is True

    def test_email_must_be_unique(self, db_session):
        u1 = User(email="dup@test.com")
        u1.set_password("x")
        db_session.add(u1)
        db_session.commit()

        u2 = User(email="dup@test.com")
        u2.set_password("y")
        db_session.add(u2)
        with pytest.raises(Exception):
            db_session.commit()


class TestRepoScanRelationships:
    def _make_user(self, db_session, email="owner@test.com"):
        user = User(email=email)
        user.set_password("x")
        db_session.add(user)
        db_session.commit()
        return user

    def test_repo_belongs_to_user(self, db_session):
        user = self._make_user(db_session)
        repo = Repo(user_id=user.id, target="https://github.com/x/y", name="y")
        db_session.add(repo)
        db_session.commit()
        assert repo.owner.email == "owner@test.com"
        assert repo in user.repos

    def test_scan_frequency_defaults_to_off(self, db_session):
        user = self._make_user(db_session)
        repo = Repo(user_id=user.id, target="https://github.com/x/y", name="y")
        db_session.add(repo)
        db_session.commit()
        assert repo.scan_frequency == "off"
        assert repo.next_scheduled_at is None

    def test_scan_belongs_to_repo_and_user(self, db_session):
        user = self._make_user(db_session)
        repo = Repo(user_id=user.id, target="https://github.com/x/y", name="y")
        db_session.add(repo)
        db_session.commit()

        scan = Scan(id="scan-1", user_id=user.id, repo_id=repo.id, status="running")
        db_session.add(scan)
        db_session.commit()

        assert scan.repo.name == "y"
        assert scan.owner.email == "owner@test.com"
        assert scan in repo.scans

    def test_deleting_user_cascades_to_repos_and_scans(self, db_session):
        user = self._make_user(db_session)
        repo = Repo(user_id=user.id, target="https://github.com/x/y", name="y")
        db_session.add(repo)
        db_session.commit()
        scan = Scan(id="scan-1", user_id=user.id, repo_id=repo.id, status="done")
        db_session.add(scan)
        db_session.commit()

        db_session.delete(user)
        db_session.commit()

        assert Repo.query.count() == 0
        assert Scan.query.count() == 0

    def test_deleting_repo_cascades_to_its_scans_only(self, db_session):
        user = self._make_user(db_session)
        repo1 = Repo(user_id=user.id, target="https://github.com/x/y", name="y")
        repo2 = Repo(user_id=user.id, target="https://github.com/x/z", name="z")
        db_session.add_all([repo1, repo2])
        db_session.commit()

        scan1 = Scan(id="scan-1", user_id=user.id, repo_id=repo1.id, status="done")
        scan2 = Scan(id="scan-2", user_id=user.id, repo_id=repo2.id, status="done")
        db_session.add_all([scan1, scan2])
        db_session.commit()

        db_session.delete(repo1)
        db_session.commit()

        remaining = Scan.query.all()
        assert len(remaining) == 1
        assert remaining[0].id == "scan-2"

    def test_scans_ordered_newest_first(self, db_session):
        import time
        user = self._make_user(db_session)
        repo = Repo(user_id=user.id, target="https://github.com/x/y", name="y")
        db_session.add(repo)
        db_session.commit()

        first = Scan(id="scan-old", user_id=user.id, repo_id=repo.id, status="done")
        db_session.add(first)
        db_session.commit()
        time.sleep(0.01)
        second = Scan(id="scan-new", user_id=user.id, repo_id=repo.id, status="done")
        db_session.add(second)
        db_session.commit()

        db_session.refresh(repo)
        assert repo.scans[0].id == "scan-new"
