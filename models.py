"""
models.py
---------
SQLAlchemy models backing the multi-user version of vuln-agent.

Schema:
    User (1) --- has many ---> Repo (personal repos, organization_id=NULL)
    User (1) --- Membership (role) ---> Organization (1) --- has many ---> Repo (team repos)
    Repo (1) --- has many ---> Scan

Two independent access-control layers, on purpose:

- User.role ("admin" | "member"): GLOBAL platform-admin flag. Answers
  "can this person see/manage things across the entire system, across
  every organization?" Used for things like /admin/scans. Unrelated to
  which organizations someone belongs to or what role they hold there.

- Membership.role ("owner" | "admin" | "member"): PER-ORGANIZATION role.
  Answers "within this specific team, what can this person do?" Someone
  can be "owner" of Org A and just "member" of Org B -- scoped entirely
  to that one organization.

Individual users and organizations coexist as first-class citizens:
- Repo.organization_id IS NULL  -> a personal repo, visible only to its
  creator (Repo.user_id), exactly like the original single-user design.
- Repo.organization_id IS SET   -> a team repo, visible to every member
  of that organization regardless of who originally added it, UNLESS
  Repo.restricted is True (see RepoAccess below).

Fine-grained repo access on top of org membership:
- Repo.restricted (bool): when True, plain "member"-role users need an
  explicit RepoAccess row to even see the repo. Org owners/admins always
  see and can push every repo in their org regardless of this flag --
  restriction narrows what plain members can reach, it never narrows an
  owner/admin's own access.
- RepoAccess(repo_id, user_id, can_push): a row grants that user
  visibility into a restricted repo; can_push additionally grants the
  right to trigger PR creation on that specific repo. can_push grants
  are meaningful even on unrestricted repos (e.g. "everyone can see
  this repo, but only these two people can actually open PRs on it").

Pending invites (OrgInvite): inviting an email with no matching User
yet creates a pending invite instead of failing. When someone signs up
with that email, matching invites are consumed automatically into real
Membership rows (see signup() in web_app.py).

DB target is controlled entirely by SQLALCHEMY_DATABASE_URI (Postgres in
every environment now -- SQLite has been retired, see web_app.py).
Schema changes are tracked with Flask-Migrate (Alembic); see migrations/.
"""

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()
migrate = Migrate()


def _utcnow():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    # Nullable as of OAuth support: a user who only ever signs in via
    # Google/GitHub has no password at all. has_password() below is the
    # single place that checks this -- use it instead of `is not None`
    # directly, so a future rename can't silently break every call site.
    password_hash = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    # --- RBAC ---
    # "admin" or "member". The very first user to sign up is promoted to
    # admin automatically (see signup() in web_app.py) so there's always
    # at least one admin account without needing a manual DB edit or a
    # separate bootstrap script.
    role = db.Column(db.String(16), default="member", nullable=False)

    repos = db.relationship(
        "Repo", backref="owner", lazy=True, cascade="all, delete-orphan"
    )
    scans = db.relationship(
        "Scan", backref="owner", lazy=True, cascade="all, delete-orphan"
    )
    memberships = db.relationship(
        "Membership", backref="user", lazy=True, cascade="all, delete-orphan"
    )
    oauth_identities = db.relationship(
        "OAuthIdentity", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def has_password(self) -> bool:
        """False for an OAuth-only account (signed up via Google/GitHub
        and never set a password). Check this before check_password()
        rather than assuming password_hash is always set."""
        return self.password_hash is not None

    def check_password(self, raw_password: str) -> bool:
        if not self.has_password():
            return False
        return check_password_hash(self.password_hash, raw_password)

    def is_admin(self) -> bool:
        """Global platform-admin check -- unrelated to org membership."""
        return self.role == "admin"

    def role_in_org(self, organization_id) -> str | None:
        """Returns this user's role ('owner'/'admin'/'member') within a
        specific organization, or None if they aren't a member at all."""
        m = Membership.query.filter_by(user_id=self.id, organization_id=organization_id).first()
        return m.role if m else None

    def organizations(self):
        """All organizations this user belongs to, regardless of role."""
        return [m.organization for m in self.memberships]

    def __repr__(self):
        return f"<User {self.email}>"


class OAuthIdentity(db.Model):
    """Links a User to one external identity provider (Google/GitHub).
    A single User can have multiple identities (e.g. signed up with a
    password, later also linked Google) -- this is a separate table
    rather than columns on User for exactly that reason.

    Matching logic (see web_app.py's oauth callback route):
      1. provider + provider_user_id already has a row -> log in as
         that row's user (fastest path, works even if the user later
         changes their email with the provider).
      2. No row, but a User already exists with the same email -> link
         this identity to that existing account (so someone who signed
         up with a password can also log in via Google afterward,
         without ending up with two separate accounts).
      3. No row and no matching email -> create a brand-new User (with
         no password -- see User.has_password()) and this identity.
    """
    __tablename__ = "oauth_identities"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    provider = db.Column(db.String(32), nullable=False)          # "google" | "github"
    provider_user_id = db.Column(db.String(255), nullable=False)  # stable id from the provider (NOT email -- emails can change)
    email = db.Column(db.String(255), nullable=True)              # email reported by the provider at link time, for display/debugging only

    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("provider", "provider_user_id", name="uq_oauth_provider_identity"),
    )

    def __repr__(self):
        return f"<OAuthIdentity {self.provider}:{self.provider_user_id} user_id={self.user_id}>"


class GmailConnection(db.Model):
    """A user's consent to let vuln-agent send email through their own
    Gmail account -- separate from OAuthIdentity/login on purpose.
    Login only ever needs a short-lived access token while the user is
    actively signing in; sending a report later, possibly minutes after
    the user has closed the tab, needs a REFRESH token (offline access)
    plus the sensitive gmail.send scope, which Google treats as a
    distinct consent from basic sign-in. Bundling that consent into the
    login flow would mean every login (even ones that never touch
    reports) prompts for an email-sending permission, which is both
    confusing and unnecessary.

    refresh_token is stored ENCRYPTED (see tools/emailer.py's encrypt/
    decrypt helpers, using GMAIL_TOKEN_ENCRYPTION_KEY) -- it is a
    long-lived credential equivalent to a password for this one scope,
    so it must never be stored or logged in plaintext.
    """
    __tablename__ = "gmail_connections"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True, index=True)
    # unique=True: one Gmail connection per user, not per repo/org -- if
    # they reconnect, the row is updated in place (see web_app.py).

    gmail_address = db.Column(db.String(255), nullable=False)
    # Which Gmail account this is, for display on a "Connected as
    # x@gmail.com [Disconnect]" settings row -- purely informational,
    # not used for auth (provider_user_id equivalent isn't needed here
    # since Google's token itself scopes access to exactly this account).

    encrypted_refresh_token = db.Column(db.Text, nullable=False)

    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    user = db.relationship("User", backref=db.backref("gmail_connection", uselist=False, cascade="all, delete-orphan"))

    def __repr__(self):
        return f"<GmailConnection user_id={self.user_id} gmail_address={self.gmail_address}>"


class SlackConnection(db.Model):
    """A user's connected Slack Incoming Webhook -- much simpler than
    GmailConnection since Slack webhooks need no OAuth dance at all:
    the user creates one directly in their own Slack workspace and
    pastes the URL in. The webhook URL itself is the entire credential
    (equivalent to a password for "post to this one channel"), so it's
    encrypted at rest the same way Gmail's refresh token is -- see
    tools/crypto_utils.py, which both now share.
    """
    __tablename__ = "slack_connections"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True, index=True)

    encrypted_webhook_url = db.Column(db.Text, nullable=False)
    # Slack doesn't hand back a stable "which channel is this" identifier
    # the way Google hands back an email address -- so this is just
    # whatever label the user typed in for their own reference (e.g.
    # "#security-alerts"), purely cosmetic, never used for auth or
    # webhook targeting.
    channel_label = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    user = db.relationship("User", backref=db.backref("slack_connection", uselist=False, cascade="all, delete-orphan"))

    def __repr__(self):
        return f"<SlackConnection user_id={self.user_id} channel_label={self.channel_label!r}>"


class Repo(db.Model):
    __tablename__ = "repos"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    # Who originally added this repo. Always set, even for team repos --
    # useful for "added by" display -- but NOT what controls visibility
    # once organization_id is set (see can_be_accessed_by below).

    organization_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=True, index=True)
    # NULL       -> personal repo, visible only to `user_id` (original,
    #               single-user behavior, unchanged).
    # NOT NULL   -> team repo, visible to every member of that
    #               organization regardless of who added it.

    target = db.Column(db.String(1024), nullable=False)   # GitHub URL or local path
    name = db.Column(db.String(255), nullable=False)       # friendly display name
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    # --- Scheduling ---
    # scan_frequency: "off" | "daily" | "weekly". When not "off", the
    # background scheduler (see web_app.py) auto-triggers a re-scan once
    # next_scheduled_at is in the past, then advances it to the next
    # occurrence. Scheduled runs auto-approve the report checkpoint but
    # NEVER auto-approve the PR checkpoint -- a human always reviews
    # fixes before anything is opened as a real PR.
    scan_frequency = db.Column(db.String(16), default="off", nullable=False)
    next_scheduled_at = db.Column(db.DateTime, nullable=True)

    # --- Fine-grained access (org repos only; ignored for personal repos) ---
    # When True, plain "member"-role org users need an explicit RepoAccess
    # row to see this repo at all. Owners/admins are never affected by
    # this flag -- they always have full access to every repo in their org.
    restricted = db.Column(db.Boolean, default=False, nullable=False)

    scans = db.relationship(
        "Scan", backref="repo", lazy=True, cascade="all, delete-orphan",
        order_by="desc(Scan.created_at)",
    )
    access_grants = db.relationship(
        "RepoAccess", backref="repo", lazy=True, cascade="all, delete-orphan"
    )

    def is_personal(self) -> bool:
        return self.organization_id is None

    def _org_role(self, user):
        return user.role_in_org(self.organization_id)

    def can_be_accessed_by(self, user) -> bool:
        """Single source of truth for repo visibility -- use this
        instead of hand-rolling a user_id/organization_id check in each
        route, so the personal-vs-team distinction can't be gotten
        wrong or forgotten in a new route later."""
        if self.is_personal():
            return self.user_id == user.id

        role = self._org_role(user)
        if role is None:
            return False
        if role in ("owner", "admin"):
            return True
        # plain member: unrestricted repos are visible to the whole org;
        # restricted repos need an explicit grant
        if not self.restricted:
            return True
        return RepoAccess.query.filter_by(repo_id=self.id, user_id=user.id).first() is not None

    def can_be_pushed_by(self, user) -> bool:
        """Whether this user may trigger PR creation on this repo.
        Separate from (and narrower than) can_be_accessed_by -- being
        able to see/scan a repo doesn't automatically mean being able
        to open a real PR against it."""
        if self.is_personal():
            return self.user_id == user.id

        role = self._org_role(user)
        if role is None:
            return False
        if role in ("owner", "admin"):
            return True
        grant = RepoAccess.query.filter_by(repo_id=self.id, user_id=user.id).first()
        return grant is not None and grant.can_push

    def __repr__(self):
        scope = f"org_id={self.organization_id}" if self.organization_id else f"user_id={self.user_id}"
        return f"<Repo {self.name} ({scope})>"


class RepoAccess(db.Model):
    """Explicit per-user grant on a single org repo. Presence of a row
    grants visibility into a restricted repo; can_push additionally
    grants the right to trigger PR creation on that repo. Irrelevant
    for personal repos (organization_id IS NULL), which are governed
    purely by Repo.user_id."""
    __tablename__ = "repo_access"

    id = db.Column(db.Integer, primary_key=True)
    repo_id = db.Column(db.Integer, db.ForeignKey("repos.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    can_push = db.Column(db.Boolean, default=False, nullable=False)

    granted_by_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    user = db.relationship("User", foreign_keys=[user_id])

    __table_args__ = (
        db.UniqueConstraint("repo_id", "user_id", name="uq_repo_access_repo_user"),
    )

    def __repr__(self):
        return f"<RepoAccess repo_id={self.repo_id} user_id={self.user_id} can_push={self.can_push}>"


class Organization(db.Model):
    __tablename__ = "organizations"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    # --- Org-level settings ---
    # default_scan_frequency: applied to new repos added to this org (as
    # a starting value only -- an individual repo's own scan_frequency
    # can still be changed afterward same as before; this just saves
    # having to set it by hand every time for teams that always want,
    # say, weekly scans on everything).
    default_scan_frequency = db.Column(db.String(16), default="off", nullable=False)

    memberships = db.relationship(
        "Membership", backref="organization", lazy=True, cascade="all, delete-orphan"
    )
    repos = db.relationship(
        "Repo", backref="organization", lazy=True, cascade="all, delete-orphan"
    )
    invites = db.relationship(
        "OrgInvite", backref="organization", lazy=True, cascade="all, delete-orphan"
    )

    def members(self):
        return [m.user for m in self.memberships]

    def owners(self):
        return [m.user for m in self.memberships if m.role == "owner"]

    def __repr__(self):
        return f"<Organization {self.name}>"


class OrgInvite(db.Model):
    """A pending invite to someone who doesn't have an account yet.
    Inviting an email with no matching User creates one of these
    instead of failing outright; signup() consumes any matching
    invites (case-insensitive email match) into real Membership rows
    the moment that person creates their account."""
    __tablename__ = "org_invites"

    id = db.Column(db.Integer, primary_key=True)
    organization_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=False, index=True)
    email = db.Column(db.String(255), nullable=False, index=True)
    role = db.Column(db.String(16), default="member", nullable=False)

    invited_by_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("organization_id", "email", name="uq_org_invite_org_email"),
    )

    def __repr__(self):
        return f"<OrgInvite org_id={self.organization_id} email={self.email} role={self.role}>"


class Membership(db.Model):
    """Join table between User and Organization, carrying the
    per-organization role. A user can be 'owner' of one org and just
    'member' of another -- this table, not User.role, is what governs
    that. See the module docstring for the full owner/admin/member
    permission model.
    """
    __tablename__ = "memberships"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    organization_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=False, index=True)

    role = db.Column(db.String(16), default="member", nullable=False)
    # "owner"  -- full control: rename/delete the org, add/remove any
    #             member (including other owners), change anyone's role.
    # "admin"  -- can add/remove repos and invite/remove "member"-level
    #             users, but cannot remove owners or delete the org.
    # "member" -- can view and scan the org's repos, cannot manage
    #             membership or org settings.

    joined_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("user_id", "organization_id", name="uq_membership_user_org"),
    )

    def __repr__(self):
        return f"<Membership user_id={self.user_id} org_id={self.organization_id} role={self.role}>"


class Scan(db.Model):
    __tablename__ = "scans"

    # Keep this a string so existing scan_id values (8-char uuids) and
    # any external references stay compatible; new rows can still use
    # uuid4 hex strings as the primary key instead of an autoincrement int.
    id = db.Column(db.String(36), primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    repo_id = db.Column(db.Integer, db.ForeignKey("repos.id"), nullable=False, index=True)

    status = db.Column(db.String(32), default="running", nullable=False)
    # running | waiting_approval | done | error

    triggered_by = db.Column(db.String(16), default="manual", nullable=False)
    # "manual" (user clicked New scan / Re-scan) or "scheduled" (background scheduler)

    report_path = db.Column(db.String(1024), nullable=True)
    sbom_path = db.Column(db.String(1024), nullable=True)
    pr_url = db.Column(db.String(1024), nullable=True)
    error = db.Column(db.Text, nullable=True)

    # Small summary stats, populated once the scan finishes -- avoids
    # re-parsing the report file just to show counts in a history list.
    code_findings_count = db.Column(db.Integer, nullable=True)
    dep_findings_count = db.Column(db.Integer, nullable=True)
    code_fixes_count = db.Column(db.Integer, nullable=True)
    withheld_fixes_count = db.Column(db.Integer, nullable=True)

    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f"<Scan {self.id} status={self.status}>"


def init_db(app):
    """Call once from web_app.py after creating the Flask app:
        from models import db, init_db
        init_db(app)

    Schema is managed entirely by Flask-Migrate/Alembic -- SQLite has
    been retired as of the organization/multi-tenancy model, so there is
    no create_all() fallback of any kind anymore. Every environment
    (including a brand-new local dev setup) must run:
        flask db init       # once, ever, to create migrations/
        flask db migrate -m "message"
        flask db upgrade     # apply pending migrations -- REQUIRED
                              # before the app will have any tables at all.

    This is deliberate: a create_all()/Alembic race on a fresh database
    is exactly what caused a `psycopg2.errors.DuplicateColumn` error the
    first time this was tried against Postgres. Removing create_all()
    entirely (rather than special-casing it for SQLite) means that race
    can't come back the next time the schema changes.
    """
    db.init_app(app)
    migrate.init_app(app, db)