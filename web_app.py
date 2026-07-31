"""
web_app.py
----------
Flask web server that serves the frontend UI and runs the vulnerability agent.

Routes:
  GET  /              → serves the UI
  POST /scan          → starts a scan, returns scan_id
  GET  /stream/<id>   → Server-Sent Events stream for live progress
  GET  /report/<id>   → serves the finished HTML report
  GET  /status/<id>   → returns current scan status as JSON
"""

import os
import sys
import time
import uuid
import json
import threading
import zipfile
import shutil
import tempfile
from functools import wraps
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, Response, send_file, render_template_string, redirect, url_for, flash
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import func

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from authlib.integrations.flask_client import OAuth

from tools.emailer import send_report_email, EmailSendError
from tools.slack_notifier import send_scan_summary, encrypt_webhook_url, SlackSendError
from tools.sbom_generator import generate_cyclonedx_sbom

from langgraph.types import Command
from models import db, init_db, User, Repo, Scan, Organization, Membership, RepoAccess, OrgInvite, OAuthIdentity, GmailConnection, SlackConnection
from logging_config import get_logger, bind_scan_id, clear_scan_id

logger = get_logger(__name__)

# --- Prometheus metrics ------------------------------------------------------
# Always registered (no env-var gate, unlike Sentry) since collecting
# metrics is essentially free and has no external dependency to be
# missing -- only the DECISION to actually scrape /metrics with a real
# Prometheus server is optional (see docker-compose.yml).
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

SCANS_TOTAL = Counter(
    "vuln_agent_scans_total", "Total scans, by final status", ["status", "triggered_by"]
)
SCAN_DURATION_SECONDS = Histogram(
    "vuln_agent_scan_duration_seconds", "Wall-clock time from scan start to finish (any outcome)"
)
CODE_FINDINGS_TOTAL = Counter(
    "vuln_agent_code_findings_total", "Static-analysis findings across all scans"
)
DEP_FINDINGS_TOTAL = Counter(
    "vuln_agent_dep_findings_total", "Vulnerable-dependency findings across all scans"
)
FIXES_TOTAL = Counter(
    "vuln_agent_fixes_total", "AI-generated fixes, by verification outcome", ["outcome"]
    # outcome: "confirmed" (passed every verification layer) | "withheld"
)
REPORT_EMAILS_TOTAL = Counter(
    "vuln_agent_report_emails_total", "Gmail auto-send attempts, by outcome", ["outcome"]
    # outcome: "sent" | "failed" | "skipped_not_connected"
)
SLACK_NOTIFICATIONS_TOTAL = Counter(
    "vuln_agent_slack_notifications_total", "Slack webhook post attempts, by outcome", ["outcome"]
    # outcome: "sent" | "failed" | "skipped_not_connected"
)

# --- Sentry (error tracking) -------------------------------------------------
# Purely opt-in via env var -- if SENTRY_DSN isn't set, sentry_sdk.init()
# is simply never called and the app behaves exactly as before. This
# mirrors the pattern used for GOOGLE_CLIENT_ID/GITHUB_CLIENT_ID above:
# missing config means "this integration is off", never a crash.
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[FlaskIntegration()],
        # traces_sample_rate: fraction of requests that get full
        # performance tracing (timing breakdowns), separate from error
        # reporting (which always fires regardless of this number). 0.2
        # is a reasonable default for a low-traffic student/portfolio
        # deployment -- turn it up if you want more granular timing
        # data, down (or to 0) to reduce Sentry quota usage.
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.2")),
        environment=os.environ.get("SENTRY_ENVIRONMENT", "development"),
    )
    logger.info("Sentry error tracking enabled")
else:
    logger.info("SENTRY_DSN not set -- Sentry error tracking disabled")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB upload limit

# --- Database + auth setup -------------------------------------------------
# SECRET_KEY signs the session cookie -- required for flask_login to work.
# Falls back to a dev-only default so the app doesn't crash if unset, but
# this MUST be a real random value (set via env var) before this is ever
# exposed beyond localhost.
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-only-insecure-key-change-me")

# Postgres only -- SQLite has been retired. DATABASE_URL is required; the
# app deliberately fails fast at startup rather than silently falling back
# to a SQLite file, since that silent fallback is exactly what caused
# confusion earlier (a scan looking "missing" because it landed in whichever
# DB happened to be the fallback in that terminal session, not the one you
# thought you were using).
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. vuln-agent requires Postgres -- SQLite is "
        "no longer supported. Set DATABASE_URL, e.g.:\n"
        "  postgresql://user:password@host:5432/vuln_agent\n"
        "(requires psycopg2-binary -- see requirements.txt). Add it to a "
        ".env file in the project root so it's picked up automatically."
    )
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

init_db(app)

# Register the DB-backed collector so /metrics also reports scan
# history recomputed live from Postgres (survives this process
# restarting -- see tools/db_metrics_collector.py for why that's
# not true of the in-memory Counters/Histograms below).
from tools.db_metrics_collector import DBBackedCollector
REGISTRY.register(DBBackedCollector(app))

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# --- Social login (OAuth) ---------------------------------------------------
# Both providers are optional independently of each other -- if a given
# provider's env vars aren't set, its client is simply never registered
# and /auth/<provider> 404s (see oauth_login below) rather than crashing
# the whole app at import time. This lets a dev enable just GitHub, just
# Google, both, or neither, without touching code.
oauth = OAuth(app)

if os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"):
    oauth.register(
        name="google",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if os.environ.get("GITHUB_CLIENT_ID") and os.environ.get("GITHUB_CLIENT_SECRET"):
    oauth.register(
        name="github",
        client_id=os.environ["GITHUB_CLIENT_ID"],
        client_secret=os.environ["GITHUB_CLIENT_SECRET"],
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "read:user user:email"},
    )


def _oauth_userinfo(provider: str, token: dict) -> tuple[str, str, str]:
    """Normalizes each provider's very different user-info shape into
    (provider_user_id, email, display_name). Raises ValueError if the
    provider didn't give us a usable, verified email -- GitHub in
    particular can have a primary email that's private, which needs a
    second API call to a dedicated endpoint (the main /user response
    omits it entirely rather than returning null)."""
    client = oauth.create_client(provider)

    if provider == "google":
        userinfo = token.get("userinfo") or client.userinfo(token=token)
        email = userinfo.get("email")
        if not email or not userinfo.get("email_verified"):
            raise ValueError("Google account has no verified email")
        return str(userinfo["sub"]), email, userinfo.get("name") or email

    if provider == "github":
        profile = client.get("user", token=token).json()
        email = profile.get("email")
        if not email:
            # Primary email can be private -- ask the emails endpoint
            # instead and pick the verified primary one.
            emails = client.get("user/emails", token=token).json()
            primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
            email = primary["email"] if primary else None
        if not email:
            raise ValueError("GitHub account has no verified email accessible")
        return str(profile["id"]), email, profile.get("login") or email

    raise ValueError(f"Unknown OAuth provider: {provider}")


def _oauth_flags() -> dict:
    """Which provider buttons AUTH_UI should show -- based on which
    clients actually got registered above, not just whether the route
    exists, so a misconfigured/missing env var hides the button instead
    of showing a button that 404s."""
    return {
        "google_enabled": "google" in oauth._clients,
        "github_enabled": "github" in oauth._clients,
    }


@app.route("/auth/<provider>")
def oauth_login(provider):
    if provider not in ("google", "github") or provider not in oauth._clients:
        return jsonify({"error": f"OAuth provider '{provider}' is not configured"}), 404
    redirect_uri = url_for("oauth_callback", provider=provider, _external=True)
    return oauth.create_client(provider).authorize_redirect(redirect_uri)


@app.route("/auth/<provider>/callback")
def oauth_callback(provider):
    if provider not in ("google", "github") or provider not in oauth._clients:
        return jsonify({"error": f"OAuth provider '{provider}' is not configured"}), 404

    client = oauth.create_client(provider)
    token = client.authorize_access_token()

    try:
        provider_user_id, email, _display_name = _oauth_userinfo(provider, token)
    except ValueError as e:
        return render_template_string(AUTH_UI, mode="login", error=str(e, **_oauth_flags()))

    email = email.strip().lower()

    # 1. Already-linked identity -- fastest path, and the only one that's
    #    correct if the person's email changed with the provider since.
    identity = OAuthIdentity.query.filter_by(provider=provider, provider_user_id=provider_user_id).first()
    if identity is not None:
        login_user(identity.user)
        return redirect(url_for("index"))

    # 2. No identity yet, but an account with this email already exists
    #    (e.g. originally signed up with a password) -- link instead of
    #    creating a duplicate account.
    user = User.query.filter_by(email=email).first()
    if user is None:
        # 3. Brand-new user -- no password at all (see User.has_password()).
        user = User(email=email)
        if User.query.count() == 0:
            user.role = "admin"
        db.session.add(user)
        db.session.flush()  # need user.id for both the identity row and invite consumption below

        pending_invites = OrgInvite.query.filter_by(email=email).all()
        for invite in pending_invites:
            db.session.add(Membership(user_id=user.id, organization_id=invite.organization_id, role=invite.role))
            db.session.delete(invite)

    db.session.add(OAuthIdentity(
        user_id=user.id,
        provider=provider,
        provider_user_id=provider_user_id,
        email=email,
    ))
    db.session.commit()

    login_user(user)
    return redirect(url_for("index"))


# --- Gmail auto-send (separate consent from login OAuth above) -------------
# Deliberately its own OAuth dance rather than reusing /auth/google:
# login only ever needs a short-lived access token while the user is
# actively present, but sending a report minutes later (possibly after
# they've closed the tab) needs a REFRESH token (offline access) plus
# the sensitive gmail.send scope -- Google treats that as a distinct
# consent, and bundling it into every login would prompt for
# email-sending permission even for people who never touch reports.
GMAIL_CONNECT_SCOPES = "openid email https://www.googleapis.com/auth/gmail.send"


@app.route("/connect/gmail")
@login_required
def connect_gmail():
    if "google" not in oauth._clients:
        return jsonify({"error": "Google OAuth is not configured -- GOOGLE_CLIENT_ID/SECRET missing"}), 404
    redirect_uri = url_for("connect_gmail_callback", _external=True)
    # access_type=offline -> issue a refresh token, not just an access
    # token. prompt=consent -> force Google to show the consent screen
    # and re-issue a refresh token even if this user connected before
    # (Google only hands one out on the FIRST consent otherwise, which
    # would silently break reconnection after a revoke or a lost key).
    return oauth.google.authorize_redirect(
        redirect_uri, scope=GMAIL_CONNECT_SCOPES, access_type="offline", prompt="consent"
    )


@app.route("/connect/gmail/callback")
@login_required
def connect_gmail_callback():
    client = oauth.create_client("google")
    token = client.authorize_access_token()

    refresh_token = token.get("refresh_token")
    if not refresh_token:
        # Happens if prompt=consent somehow didn't take (e.g. the
        # request was tampered with) -- without a refresh token there
        # is nothing usable to store, so bail out with a clear message
        # rather than silently saving a connection that can never send.
        flash("Gmail connection failed -- Google did not return a refresh token. Please try again.")
        return redirect(url_for("account_page"))

    userinfo = token.get("userinfo") or client.userinfo(token=token)
    gmail_address = userinfo.get("email")
    if not gmail_address:
        flash("Gmail connection failed -- could not read your Gmail address.")
        return redirect(url_for("account_page"))

    from tools.emailer import encrypt_refresh_token
    encrypted = encrypt_refresh_token(refresh_token)

    existing = GmailConnection.query.filter_by(user_id=current_user.id).first()
    if existing:
        existing.gmail_address = gmail_address
        existing.encrypted_refresh_token = encrypted
    else:
        db.session.add(GmailConnection(
            user_id=current_user.id,
            gmail_address=gmail_address,
            encrypted_refresh_token=encrypted,
        ))
    db.session.commit()

    flash(f"Gmail connected: reports will be emailed to {gmail_address}")
    return redirect(url_for("account_page"))


@app.route("/disconnect/gmail", methods=["POST"])
@login_required
def disconnect_gmail():
    existing = GmailConnection.query.filter_by(user_id=current_user.id).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
    return redirect(url_for("account_page"))


@app.route("/connect/slack", methods=["POST"])
@login_required
def connect_slack():
    """No OAuth needed -- the user creates their own Incoming Webhook
    directly in Slack (Slack API site -> Apps -> Incoming Webhooks) and
    pastes the URL here. Sends an immediate test message on save so the
    user gets instant confirmation it's wired up correctly, rather than
    finding out only whenever their next scan happens to finish."""
    webhook_url = (request.form.get("webhook_url") or "").strip()
    channel_label = (request.form.get("channel_label") or "").strip() or None

    if not webhook_url.startswith("https://hooks.slack.com/"):
        flash("That doesn't look like a Slack Incoming Webhook URL (should start with https://hooks.slack.com/).")
        return redirect(url_for("account_page"))

    encrypted = encrypt_webhook_url(webhook_url)

    try:
        send_scan_summary(
            encrypted, repo_name="(test)", scan_id="connection-test",
            code_findings=0, dep_findings=0, fixes_confirmed=0, fixes_withheld=0,
        )
    except SlackSendError as e:
        flash(f"Couldn't send a test message to that webhook -- double check the URL. ({e})")
        return redirect(url_for("account_page"))

    existing = SlackConnection.query.filter_by(user_id=current_user.id).first()
    if existing:
        existing.encrypted_webhook_url = encrypted
        existing.channel_label = channel_label
    else:
        db.session.add(SlackConnection(user_id=current_user.id, encrypted_webhook_url=encrypted, channel_label=channel_label))
    db.session.commit()

    flash("Slack connected -- check your channel for a test message.")
    return redirect(url_for("account_page"))


@app.route("/disconnect/slack", methods=["POST"])
@login_required
def disconnect_slack():
    existing = SlackConnection.query.filter_by(user_id=current_user.id).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
    return redirect(url_for("account_page"))


@app.route("/account")
@login_required
def account_page():
    gmail_connection = GmailConnection.query.filter_by(user_id=current_user.id).first()
    slack_connection = SlackConnection.query.filter_by(user_id=current_user.id).first()
    return render_template_string(ACCOUNT_UI, connection=gmail_connection, slack_connection=slack_connection, **_oauth_flags())


def require_role(role: str):
    """Route decorator for RBAC. Use below @login_required (it assumes
    current_user is already authenticated) on any route that should be
    restricted to a specific role -- e.g.:

        @app.route("/admin/scans")
        @login_required
        @require_role("admin")
        def admin_all_scans(): ...

    403s rather than redirecting, since an authenticated user hitting a
    role-gated route they don't have access to is a permissions problem,
    not a "please log in" problem.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if current_user.role != role:
                return jsonify({"error": "Forbidden -- requires role: " + role}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# --- Org-aware access control -----------------------------------------------
# Roles are TWO SEPARATE things in this app, deliberately:
#   - User.role ("admin"/"member")      -- global platform-admin flag
#   - Membership.role ("owner"/"admin"/"member") -- per-organization role
# require_role() above only checks the first. Everything below checks the
# second, scoped to one specific organization (via a URL's <org_id>).

ORG_ROLE_RANK = {"member": 0, "admin": 1, "owner": 2}


def require_org_role(min_role: str):
    """Route decorator for a route like /organizations/<org_id>/... --
    requires the URL to have an `org_id` kwarg, and checks the current
    user's MEMBERSHIP role in that specific org (not their global
    User.role) meets or exceeds min_role in the owner > admin > member
    hierarchy. 404s (not 403) if they aren't a member at all, so a
    non-member can't even confirm the org exists.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            org_id = kwargs.get("org_id")
            role = current_user.role_in_org(org_id)
            if role is None:
                return jsonify({"error": "Organization not found"}), 404
            if ORG_ROLE_RANK.get(role, -1) < ORG_ROLE_RANK.get(min_role, 999):
                return jsonify({"error": f"Forbidden -- requires org role: {min_role} or higher"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def get_accessible_repo_or_404(repo_id):
    """Single place every route uses to fetch a Repo the current user is
    actually allowed to see -- personal (user_id match) or team (org
    membership, any role). Replaces a plain `Repo.query.get(repo_id)`,
    which would otherwise let any logged-in user view/modify any repo by
    guessing its id.
    """
    repo = db.session.get(Repo, repo_id)
    if repo is None or not repo.can_be_accessed_by(current_user):
        return None
    return repo


def get_user_accessible_repos(user):
    """Every repo the user can see: their own personal repos, PLUS every
    repo belonging to any organization they're a member of (any role),
    minus any org repos marked `restricted` that this user hasn't been
    explicitly granted access to (see Repo.can_be_accessed_by).
    Used by /repos and /history so team repos show up alongside personal
    ones instead of only ever showing what one person added themselves.
    """
    org_ids = [m.organization_id for m in user.memberships]
    query = Repo.query.filter(
        db.or_(
            db.and_(Repo.user_id == user.id, Repo.organization_id.is_(None)),
            Repo.organization_id.in_(org_ids) if org_ids else db.false(),
        )
    )
    repos = query.order_by(Repo.created_at.desc()).all()
    return [r for r in repos if r.can_be_accessed_by(user)]


def can_access_in_memory_scan(scan_dict) -> bool:
    """Org-aware replacement for `scan.get("user_id") == current_user.id`
    on the in-memory SCANS dict entries. Falls back to the plain user_id
    check if repo_id somehow isn't present (shouldn't happen for scans
    launched after this change, but keeps old in-flight scans working
    across a deploy)."""
    repo_id = scan_dict.get("repo_id")
    if repo_id is None:
        return scan_dict.get("user_id") == current_user.id
    repo = db.session.get(Repo, repo_id)
    return repo is not None and repo.can_be_accessed_by(current_user)


def can_access_scan_row(scan_row: Scan) -> bool:
    """Org-aware replacement for `Scan.query.filter_by(user_id=...)` when
    working with a persisted Scan row -- checks access via the scan's
    Repo (personal or org), not who happened to trigger the run."""
    return scan_row is not None and scan_row.repo is not None and scan_row.repo.can_be_accessed_by(current_user)

# In-memory store for scan state
# { scan_id: { "status": "running"|"waiting_approval"|"done"|"error", "events": [...], "report_path": str } }
SCANS = {}
SCANS_LOCK = threading.Lock()


def _maybe_notify_slack(scan_row, repo_name: str, code_count: int, dep_count: int,
                         confirmed_count: int, withheld_count: int, pr_url: str | None) -> None:
    """Called from inside an existing app_context (see the call site in
    run_scan_thread) -- unlike _maybe_email_report, this does NOT open
    its own app_context, since the caller already has scan_row loaded
    and committed. Silently does nothing if the repo owner hasn't
    connected Slack; any send failure is caught and logged, never
    raised -- same non-blocking principle as Gmail auto-send."""
    connection = SlackConnection.query.filter_by(user_id=scan_row.user_id).first()
    if connection is None:
        SLACK_NOTIFICATIONS_TOTAL.labels(outcome="skipped_not_connected").inc()
        return

    try:
        report_url = url_for("serve_report", scan_id=scan_row.id, _external=True)
    except Exception:
        report_url = None

    try:
        send_scan_summary(
            connection.encrypted_webhook_url,
            repo_name=repo_name,
            scan_id=scan_row.id,
            code_findings=code_count,
            dep_findings=dep_count,
            fixes_confirmed=confirmed_count,
            fixes_withheld=withheld_count,
            pr_url=pr_url,
            report_url=report_url,
        )
        SLACK_NOTIFICATIONS_TOTAL.labels(outcome="sent").inc()
    except SlackSendError as e:
        logger.warning(f"[slack] failed to notify for scan {scan_row.id}: {e}")
        SLACK_NOTIFICATIONS_TOTAL.labels(outcome="failed").inc()


def _maybe_email_report(scan_id: str, report_path: str | None) -> None:
    """Fires the moment report_node finishes (see the call site in
    run_scan_thread below) -- fully separate from agent/nodes.py on
    purpose: report_node runs OUTSIDE any Flask app context (it's a
    plain LangGraph node, also used by the CLI path in agent.py, which
    has no DB/Flask-SQLAlchemy at all), so this Gmail lookup + send has
    to live here in the web app instead of inside the node itself.

    Silently does nothing if: the repo owner hasn't connected Gmail, or
    the report file can't be found. Any actual send failure is caught
    and logged, NEVER raised -- the scan already succeeded and the
    report is still reachable at /report/<scan_id> regardless of
    whether this email goes out."""
    if not report_path or not os.path.exists(report_path):
        logger.warning(f"[email] report_path missing or not found for scan {scan_id}: {report_path!r}")
        return

    with app.app_context():
        scan_row = db.session.get(Scan, scan_id)
        if scan_row is None:
            return

        connection = GmailConnection.query.filter_by(user_id=scan_row.user_id).first()
        if connection is None:
            REPORT_EMAILS_TOTAL.labels(outcome="skipped_not_connected").inc()
            return  # repo owner hasn't connected Gmail -- nothing to do

        repo_name = scan_row.repo.name if scan_row.repo else "unknown-repo"

        try:
            with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                report_html = f.read()
        except OSError as e:
            logger.warning(f"[email] could not read report file for scan {scan_id}: {e}")
            REPORT_EMAILS_TOTAL.labels(outcome="failed").inc()
            return

        try:
            send_report_email(connection, subject=f"vuln-agent report: {repo_name}", report_html=report_html)
            REPORT_EMAILS_TOTAL.labels(outcome="sent").inc()
        except EmailSendError as e:
            logger.warning(f"[email] failed to send report for scan {scan_id}: {e}")
            REPORT_EMAILS_TOTAL.labels(outcome="failed").inc()


def run_scan_thread(scan_id: str, target: str, is_scheduled: bool = False):
    """Run the agent in a background thread, posting events as nodes complete.
    Pauses on a threading.Event whenever the graph hits interrupt() (report
    checkpoint or PR checkpoint), and resumes once the browser POSTs a
    decision to /approve/<scan_id>.

    If is_scheduled is True (this run was triggered automatically by the
    background scheduler, not a person clicking a button), the report
    checkpoint is auto-approved immediately -- but the PR checkpoint is
    ALWAYS auto-rejected regardless. A scheduled scan can produce a
    report and proposed fixes fully unattended, but opening a real PR
    always requires an actual person reviewing it first."""
    from agent.graph import build_graph

    bind_scan_id(scan_id)
    logger.info(f"Scan started: target={target!r} scheduled={is_scheduled}")

    _metrics_start_time = time.time()
    graph = build_graph()

    def post_event(event_type: str, data: dict):
        with SCANS_LOCK:
            SCANS[scan_id]["events"].append({"type": event_type, "data": data})

    post_event("start", {"message": "Scan started", "target": target})

    # Set output dir to reports folder inside project
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    os.environ["AGENT_OUTPUT_DIR"] = reports_dir
    os.environ["AGENT_QUIET"] = "1"

    NODE_LABELS = {
        "router":       "Detecting input type",
        "github_fetch": "Cloning repository",
        "local_read":   "Reading local folder",
        "code_scan":    "Running static analysis",
        "dep_scan":     "Scanning dependencies",
        "cve_enrich":   "Enriching with CVE data",
        "human_review": "Recording report decision",
        "report":       "Generating AI report",
        "fix_generate": "Generating fixes (with critic review)",
        "pr_review":    "Recording PR decision",
        "pr_generate":  "Opening pull request",
    }

    initial_state = {
        "target": target,
        "scan_id": scan_id,
        "input_type": None,
        "repo_path": None,
        "repo_name": None,
        "code_findings": None,
        "dep_findings": None,
        "enriched_findings": None,
        "human_decision": None,
        "report_markdown": None,
        "report_path": None,
        "code_fixes": None,
        "withheld_code_fixes": None,
        "dependency_fixes": None,
        "fixes_output_dir": None,
        "pr_decision": None,
        "pr_url": None,
        "error": None,
    }

    config = {"configurable": {"thread_id": scan_id}}

    try:
        accumulated = {**initial_state}
        step = 0
        total = len(NODE_LABELS)
        stream_input = initial_state

        while True:
            interrupted = False

            for event in graph.stream(stream_input, config, stream_mode="updates"):
                for node_name, node_output in event.items():

                    if node_name == "__interrupt__":
                        payload = node_output[0].value
                        checkpoint = payload.get("checkpoint")

                        if is_scheduled:
                            # No person is watching this run -- decide
                            # automatically instead of blocking on a
                            # browser POST that will never come.
                            if checkpoint == "report_approval":
                                answer = "approve"
                            else:
                                # pr_approval (or anything unexpected) --
                                # never auto-open a real PR unattended.
                                answer = "reject"

                            post_event("approval_required", payload)
                            post_event("auto_decision", {"checkpoint": checkpoint, "answer": answer})
                            logger.info(f"Scheduled scan auto-decided checkpoint={checkpoint!r}: {answer!r}")

                            with SCANS_LOCK:
                                SCANS[scan_id]["status"] = "running"

                            stream_input = Command(resume=answer)
                            interrupted = True
                            break

                        with SCANS_LOCK:
                            SCANS[scan_id]["status"] = "waiting_approval"
                            SCANS[scan_id]["checkpoint"] = checkpoint
                            SCANS[scan_id]["approval_answer"] = None
                            approval_event = SCANS[scan_id]["approval_event"]
                            approval_event.clear()

                        post_event("approval_required", payload)
                        logger.info(f"Scan paused, waiting for human approval at checkpoint={checkpoint!r}")

                        # Block this background thread until the browser
                        # POSTs a decision to /approve/<scan_id>.
                        approval_event.wait()

                        with SCANS_LOCK:
                            answer = SCANS[scan_id]["approval_answer"]
                            SCANS[scan_id]["status"] = "running"

                        logger.info(f"Checkpoint {checkpoint!r} decided: {answer!r}")

                        stream_input = Command(resume=answer)
                        interrupted = True
                        break

                    step += 1
                    accumulated.update(node_output)
                    post_event("node_complete", {
                        "node": node_name,
                        "label": NODE_LABELS.get(node_name, node_name),
                        "step": step,
                        "total": total,
                        "error": node_output.get("error")
                    })

                    if node_name == "report" and not node_output.get("error"):
                        # Fire immediately on report generation, not at
                        # scan end -- matches the earlier decision that
                        # the email shouldn't wait on the PR checkpoint,
                        # which can be minutes/hours away pending human
                        # review.
                        _maybe_email_report(scan_id, node_output.get("report_path"))

                if interrupted:
                    break

            if not interrupted:
                break

        final_state = accumulated

        if final_state.get("error"):
            logger.error(f"Scan finished with error: {final_state['error']}")
            post_event("error", {"message": final_state["error"]})
            with SCANS_LOCK:
                SCANS[scan_id]["status"] = "error"
                SCANS[scan_id]["error"] = final_state["error"]

            SCANS_TOTAL.labels(status="error", triggered_by="scheduled" if is_scheduled else "manual").inc()
            SCAN_DURATION_SECONDS.observe(time.time() - _metrics_start_time)

            with app.app_context():
                scan_row = db.session.get(Scan, scan_id)
                if scan_row:
                    scan_row.status = "error"
                    scan_row.error = final_state["error"]
                    scan_row.completed_at = datetime.now(timezone.utc)
                    db.session.commit()
        else:
            report_path = final_state.get("report_path", "")
            code_count  = len(final_state.get("code_findings") or [])
            dep_count   = len(final_state.get("dep_findings") or [])
            code_fixes  = final_state.get("code_fixes") or []
            withheld_fixes = final_state.get("withheld_code_fixes") or []
            dep_fixes   = final_state.get("dependency_fixes") or []
            pr_url      = final_state.get("pr_url")

            # SBOM generation: vulnerability-only scope -- see
            # tools/sbom_generator.py's module docstring for exactly
            # why this isn't a complete dependency inventory. Written
            # next to the HTML report (same directory, same naming
            # convention) so it survives the same way report_path does
            # -- readable later even after this thread/process ends.
            sbom_path = None
            try:
                dep_enriched = [f for f in (final_state.get("enriched_findings") or []) if f.get("finding_type") == "dependency"]
                sbom_dict = generate_cyclonedx_sbom(dep_enriched, repo_name=final_state.get("repo_name", "unknown-repo"))
                if report_path:
                    sbom_path = os.path.splitext(report_path)[0] + ".sbom.json"
                else:
                    sbom_path = os.path.join(tempfile.gettempdir(), f"sbom_{scan_id}.json")
                with open(sbom_path, "w", encoding="utf-8") as f:
                    json.dump(sbom_dict, f, indent=2)
            except Exception as e:
                # SBOM generation is an enhancement, not a required step
                # -- must never fail a scan that otherwise completed
                # successfully, same non-blocking principle as Gmail/
                # Slack.
                logger.warning(f"[sbom] failed to generate SBOM for scan {scan_id}: {e}")
                sbom_path = None

            SCANS_TOTAL.labels(status="done", triggered_by="scheduled" if is_scheduled else "manual").inc()
            SCAN_DURATION_SECONDS.observe(time.time() - _metrics_start_time)
            CODE_FINDINGS_TOTAL.inc(code_count)
            DEP_FINDINGS_TOTAL.inc(dep_count)
            FIXES_TOTAL.labels(outcome="confirmed").inc(len(code_fixes))
            FIXES_TOTAL.labels(outcome="withheld").inc(len(withheld_fixes))

            # Diff-viewer data: combine confirmed + withheld code fixes
            # into ONE indexed list, each tagged "confirmed" so the
            # frontend can badge them differently. This has to happen
            # HERE (inside the scan thread, right after the graph
            # finishes) rather than lazily when the diff route is hit
            # later -- the temp clone directory these fixes were
            # generated from gets deleted once the scan completes, and
            # more importantly the graph's in-memory state (final_state)
            # itself doesn't outlive this function call. If this isn't
            # captured now, it's gone.
            fixes_for_diff = []
            for f in code_fixes:
                fixes_for_diff.append({**f, "confirmed": True})
            for f in withheld_fixes:
                fixes_for_diff.append({**f, "confirmed": False})

            logger.info(
                f"Scan completed: {code_count} code finding(s), {dep_count} dep finding(s), "
                f"{len(code_fixes)} fix(es) confirmed, {len(withheld_fixes)} withheld, "
                f"pr_url={pr_url or 'none'}"
            )

            post_event("done", {
                "report_path": report_path,
                "code_findings": code_count,
                "dep_findings": dep_count,
                "total_findings": code_count + dep_count,
                "code_fixes_count": len(code_fixes),
                "withheld_fixes_count": len(withheld_fixes),
                "dep_fixes_count": len(dep_fixes),
                "pr_url": pr_url,
                "scan_id": scan_id
            })

            with SCANS_LOCK:
                SCANS[scan_id]["status"] = "done"
                SCANS[scan_id]["report_path"] = report_path
                SCANS[scan_id]["sbom_path"] = sbom_path
                SCANS[scan_id]["fixes"] = fixes_for_diff
                SCANS[scan_id]["stats"] = {
                    "code": code_count,
                    "dep": dep_count,
                    "total": code_count + dep_count,
                    "code_fixes": len(code_fixes),
                    "withheld_fixes": len(withheld_fixes),
                    "dep_fixes": len(dep_fixes),
                    "pr_url": pr_url,
                }

            with app.app_context():
                scan_row = db.session.get(Scan, scan_id)
                if scan_row:
                    scan_row.status = "done"
                    scan_row.report_path = report_path
                    scan_row.sbom_path = sbom_path
                    scan_row.pr_url = pr_url
                    scan_row.code_findings_count = code_count
                    scan_row.dep_findings_count = dep_count
                    scan_row.code_fixes_count = len(code_fixes)
                    scan_row.withheld_fixes_count = len(withheld_fixes)
                    scan_row.completed_at = datetime.now(timezone.utc)
                    db.session.commit()

                    # Fires here, not alongside the Gmail hook in
                    # report_node's completion -- this is the point
                    # where findings + fixes + PR are ALL known, matching
                    # the earlier decision that Slack represents "the
                    # whole run is done" rather than "the report exists"
                    # (see tools/slack_notifier.py's module docstring).
                    _maybe_notify_slack(
                        scan_row, repo_name=scan_row.repo.name if scan_row.repo else "unknown-repo",
                        code_count=code_count, dep_count=dep_count,
                        confirmed_count=len(code_fixes), withheld_count=len(withheld_fixes),
                        pr_url=pr_url,
                    )

    except Exception as e:
        # Full traceback now goes to logs/vuln_agent.log (permanent,
        # survives the terminal closing/scrolling) in addition to the
        # console -- this is exactly the kind of silent-death case that
        # used to leave zero record once the terminal history was gone.
        logger.exception(f"Unhandled exception in scan thread: {e}")
        if SENTRY_DSN:
            # Flask's FlaskIntegration only auto-captures exceptions
            # that escape an actual HTTP request -- this background
            # thread runs completely outside any request, so without
            # this explicit call, a crash here would never reach
            # Sentry at all despite being one of the most important
            # places to know about a failure.
            import sentry_sdk
            sentry_sdk.capture_exception(e)
        post_event("error", {"message": str(e)})
        with SCANS_LOCK:
            SCANS[scan_id]["status"] = "error"
            SCANS[scan_id]["error"] = str(e)

        with app.app_context():
            scan_row = db.session.get(Scan, scan_id)
            if scan_row:
                scan_row.status = "error"
                scan_row.error = str(e)
                scan_row.completed_at = datetime.now(timezone.utc)
                db.session.commit()

    finally:
        # This thread may be reused (or its identity coincidentally
        # matches a future scan's thread) -- always clear so a stale
        # scan_id can never leak into unrelated log lines later.
        clear_scan_id()


# --- Diff viewer: rendering helpers -----------------------------------------
# Deliberately a compact, single-column unified diff (not side-by-side) --
# side-by-side needs horizontal space this app can't assume it has (the UI
# is used on mobile too), and a unified view is the format most people
# reviewing a security fix are already used to reading (same as `git diff`).
import difflib
import html as html_lib


def _render_unified_diff_html(original: str, fixed: str) -> str:
    """Renders a line-level unified diff as small, theme-matching HTML
    divs (not difflib.HtmlDiff's default output, which ships its own
    ugly inline styling that would clash with this app's dark theme).
    """
    original_lines = original.splitlines()
    fixed_lines = fixed.splitlines()
    diff = difflib.unified_diff(original_lines, fixed_lines, lineterm="", n=3)

    rows = []
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            continue
        escaped = html_lib.escape(line)
        if line.startswith("@@"):
            rows.append(f'<div class="diff-hunk">{escaped}</div>')
        elif line.startswith("+"):
            rows.append(f'<div class="diff-line diff-add">{escaped}</div>')
        elif line.startswith("-"):
            rows.append(f'<div class="diff-line diff-del">{escaped}</div>')
        else:
            rows.append(f'<div class="diff-line diff-ctx">{escaped}</div>')

    if not rows:
        return '<div class="diff-line diff-ctx">(no textual differences)</div>'
    return "\n".join(rows)


def _render_verification_checklist_html(fix: dict) -> str:
    """Renders the same verification signals the HTML report already
    shows (see report_generator.py's four-state Pass/Fail/Skipped/N-A
    pattern) as compact rows for the diff viewer modal, so a reviewer
    doesn't have to cross-reference the full report to see WHY a fix
    was confirmed or withheld.
    """
    v = fix.get("verification", {}) or {}
    critique = fix.get("critique", {}) or {}
    rows = []

    def row(status: str, label: str, css_class: str):
        rows.append(f'<div class="diff-check"><span class="diff-check-badge {css_class}">{html_lib.escape(status)}</span> {html_lib.escape(label)}</div>')

    if v.get("syntax_valid", True):
        row("PASS", "Syntax check", "check-pass")
    else:
        row("FAIL", "Syntax check failed", "check-fail")

    if v.get("bandit_still_flags"):
        row("FAIL", f"Ground-truth re-scan: rule(s) {v.get('remaining_issues')} still detected", "check-fail")
    else:
        row("PASS", "Ground-truth re-scan: original rule(s) no longer detected", "check-pass")

    pyflakes = v.get("pyflakes") or {}
    if pyflakes.get("skipped"):
        row("SKIP", f"Correctness check: {pyflakes.get('skip_reason', 'not applicable')}", "check-skip")
    elif pyflakes.get("has_critical"):
        row("FAIL", "Correctness check found a real bug", "check-fail")
    elif pyflakes:
        row("PASS", "Correctness check: no issues", "check-pass")

    deleted = v.get("deleted_definitions") or []
    if deleted:
        row("FAIL", f"Deleted definitions detected: {deleted}", "check-fail")
    else:
        row("PASS", "No function/class definitions were deleted", "check-pass")

    coverage = v.get("rule_coverage") or {}
    if coverage.get("applicable") and not coverage.get("ok", True):
        row("PARTIAL", f"Critic did not confirm rule(s) {coverage.get('missing')} were resolved", "check-warn")
    elif coverage.get("applicable"):
        row("PASS", "Critic confirmed every original finding rule was addressed", "check-pass")

    existing_tests = v.get("existing_tests") or {}
    if existing_tests.get("tests_found") and existing_tests.get("tests_passed") is False:
        row("FAIL", f"Existing test suite fails: {existing_tests.get('summary', '')}", "check-fail")
    elif existing_tests.get("tests_found") and existing_tests.get("tests_passed"):
        row("PASS", "Existing test suite passes", "check-pass")
    elif existing_tests.get("skip_reason"):
        row("SKIP", existing_tests.get("skip_reason"), "check-skip")

    score = critique.get("score")
    verdict = critique.get("verdict", "unavailable")
    verdict_class = {"pass": "check-pass", "fail": "check-fail", "needs_improvement": "check-warn"}.get(verdict, "check-skip")
    score_text = f"{score}/10" if score is not None else "N/A"
    row(verdict.upper(), f"Independent critic review: {score_text}", verdict_class)

    for p in (critique.get("problems") or [])[:5]:
        text = p.get("text", p) if isinstance(p, dict) else p
        severity = p.get("severity", "moderate").upper() if isinstance(p, dict) else "MODERATE"
        rows.append(f'<div class="diff-problem"><span class="diff-problem-sev">[{html_lib.escape(severity)}]</span> {html_lib.escape(str(text))}</div>')

    return "\n".join(rows)



@app.route("/health")
def health():
    """Unauthenticated health check -- for uptime monitors, load
    balancers, or just checking the app is actually up without opening
    a browser. Verifies DB connectivity specifically (not just "the
    Flask process is running") since a DB-down app can still answer
    HTTP requests while every real feature is broken."""
    try:
        db.session.execute(db.text("SELECT 1"))
        return jsonify({"status": "ok", "db": "ok"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "db": "error", "detail": str(e)}), 503


@app.route("/")
def index():
    # Public editorial landing for logged-out visitors; authenticated users
    # go straight to their overview dashboard.
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template_string(LANDING_UI)


@app.route("/scan-console")
@login_required
def scan_console():
    org_options = "".join(
        f'<option value="{m.organization_id}">{m.organization.name}</option>'
        for m in current_user.memberships
    )
    return render_template_string(HTML_UI, org_options=org_options)


@app.route("/dashboard")
@login_required
def dashboard():
    repos = get_user_accessible_repos(current_user)

    open_findings = 0
    done_count = 0
    latest_scan_dt = None
    rows = ""
    for r in repos:
        last_scan = r.scans[0] if r.scans else None
        if last_scan:
            when = f"{last_scan.created_at:%Y-%m-%d %H:%M}"
            if latest_scan_dt is None or last_scan.created_at > latest_scan_dt:
                latest_scan_dt = last_scan.created_at
            if last_scan.status == "done":
                done_count += 1
                findings = (last_scan.code_findings_count or 0) + (last_scan.dep_findings_count or 0)
                open_findings += findings
                if findings:
                    sev_cls, sev_txt = "high", f"{findings} finding" + ("s" if findings != 1 else "")
                else:
                    sev_cls, sev_txt = "clear", "Clear"
            else:
                sev_cls, sev_txt = "", last_scan.status
        else:
            when, sev_cls, sev_txt = "never", "", "—"

        scope = r.organization.name if r.organization_id else "Personal"
        rows += f"""<div class="repo-row">
          <div><div class="name">{r.name}</div><div class="sub">{r.target} · {scope}</div></div>
          <div class="sub">{when}</div>
          <div class="sev {sev_cls}">{sev_txt}</div>
          <a class="mono-link" href="/repos">Review →</a>
        </div>"""

    last_scan_display = f"{latest_scan_dt:%b %d}" if latest_scan_dt else "—"
    return render_template_string(
        DASHBOARD_UI,
        user_email=current_user.email,
        repo_count=len(repos),
        open_findings=open_findings,
        done_count=done_count,
        last_scan=last_scan_display,
        rows=rows,
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template_string(AUTH_UI, mode="signup", error="Email and password are required.", **_oauth_flags())

        if User.query.filter_by(email=email).first():
            return render_template_string(AUTH_UI, mode="signup", error="An account with that email already exists.", **_oauth_flags())

        user = User(email=email)
        user.set_password(password)
        # First account ever created becomes admin automatically, so
        # there's always at least one admin without a manual DB edit or
        # a separate bootstrap script. Every account after that defaults
        # to "member" (see models.py).
        if User.query.count() == 0:
            user.role = "admin"
        db.session.add(user)
        db.session.flush()  # need user.id before turning invites into memberships

        # Consume any pending invites sent to this email before the
        # account existed -- turn each into a real Membership now.
        pending_invites = OrgInvite.query.filter_by(email=email).all()
        for invite in pending_invites:
            db.session.add(Membership(user_id=user.id, organization_id=invite.organization_id, role=invite.role))
            db.session.delete(invite)

        db.session.commit()

        login_user(user)
        return redirect(url_for("index"))

    return render_template_string(AUTH_UI, mode="signup", error=None, **_oauth_flags())


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return render_template_string(AUTH_UI, mode="login", error="Invalid email or password.", **_oauth_flags())

        login_user(user)
        return redirect(url_for("index"))

    return render_template_string(AUTH_UI, mode="login", error=None, **_oauth_flags())


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


def _launch_scan(repo: Repo, triggered_by: str = "manual") -> str:
    """Create a new Scan row + in-memory record for an existing Repo, and
    kick off the background thread. Shared by /scan (new target),
    /repos/<id>/scan (manual re-scan), and the background scheduler
    (automatic re-scan) so the launch logic only lives in one place.

    triggered_by: "manual" or "scheduled" -- recorded on the Scan row for
    history/auditing, and controls whether approval checkpoints block on
    a human (manual) or auto-decide (scheduled; see run_scan_thread)."""
    is_scheduled = triggered_by == "scheduled"
    scan_id = str(uuid.uuid4())[:8]

    scan_row = Scan(id=scan_id, user_id=repo.user_id, repo_id=repo.id, status="running", triggered_by=triggered_by)
    db.session.add(scan_row)
    db.session.commit()

    with SCANS_LOCK:
        SCANS[scan_id] = {
            "status": "running",
            "events": [],
            "report_path": None,
            "error": None,
            "target": repo.target,
            "user_id": repo.user_id,
            "repo_id": repo.id,
            "approval_event": threading.Event(),
            "approval_answer": None,
        }

    t = threading.Thread(target=run_scan_thread, args=(scan_id, repo.target, is_scheduled), daemon=True)
    t.start()

    return scan_id


# ---------------------------------------------------------------------------
# Scheduled (automatic) re-scans
# ---------------------------------------------------------------------------
_FREQUENCY_TO_TIMEDELTA = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
}


def _check_scheduled_scans():
    """Runs periodically in the background (see scheduler.add_job below).
    Finds every repo whose schedule is due, launches a scan for it via
    the same _launch_scan() path manual re-scans use (triggered_by
    marks it as "scheduled" so run_scan_thread auto-answers the report
    checkpoint and auto-rejects the PR checkpoint), then advances
    next_scheduled_at to the next occurrence.

    Runs inside its own app context since it's called from APScheduler's
    background thread, not from a Flask request."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        due_repos = Repo.query.filter(
            Repo.scan_frequency != "off",
            Repo.next_scheduled_at.isnot(None),
            Repo.next_scheduled_at <= now,
        ).all()

        for repo in due_repos:
            delta = _FREQUENCY_TO_TIMEDELTA.get(repo.scan_frequency)
            if delta is None:
                continue  # unknown frequency value -- skip rather than guess

            print(f"[scheduler] {repo.name} is due for a {repo.scan_frequency} re-scan -- launching")
            try:
                _launch_scan(repo, triggered_by="scheduled")
            except Exception as e:
                print(f"[scheduler] Failed to launch scheduled scan for {repo.name}: {e}")

            # Advance regardless of launch success -- a transient failure
            # to start this cycle shouldn't permanently stall the
            # schedule; it'll simply try again next cycle after that.
            repo.next_scheduled_at = now + delta
            db.session.commit()


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(_check_scheduled_scans, "interval", minutes=5, id="check_scheduled_scans")
scheduler.start()


@app.route("/repos/<int:repo_id>/schedule", methods=["POST"])
@login_required
def set_repo_schedule(repo_id):
    """Set (or turn off) a repo's auto re-scan frequency from the /repos page."""
    repo = get_accessible_repo_or_404(repo_id)
    if not repo:
        return "Repo not found", 404

    frequency = request.form.get("frequency", "off")
    if frequency not in ("off", "daily", "weekly"):
        return "Invalid frequency", 400

    repo.scan_frequency = frequency
    if frequency == "off":
        repo.next_scheduled_at = None
    else:
        # Schedule the first automatic run one interval from now, not
        # immediately -- turning on "daily" shouldn't instantly trigger
        # a scan the moment you save the setting.
        repo.next_scheduled_at = datetime.now(timezone.utc) + _FREQUENCY_TO_TIMEDELTA[frequency]

    db.session.commit()
    return redirect(url_for("repos_page"))


@app.route("/scan", methods=["POST"])
@login_required
def start_scan():
    target = None

    # GitHub URL submission
    if request.is_json:
        data = request.get_json()
        target = data.get("url", "").strip()
        if not target:
            return jsonify({"error": "No URL provided"}), 400

    # File upload submission
    elif "file" in request.files:
        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded zip to temp dir and extract
        temp_dir = tempfile.mkdtemp(prefix="vuln_upload_")
        zip_path = os.path.join(temp_dir, "upload.zip")
        uploaded.save(zip_path)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                extract_dir = os.path.join(temp_dir, "extracted")
                zf.extractall(extract_dir)
            target = extract_dir
        except zipfile.BadZipFile:
            return jsonify({"error": "Uploaded file is not a valid zip archive"}), 400
    else:
        return jsonify({"error": "No input provided"}), 400

    # Optional: attach this repo to an organization instead of keeping it
    # personal. Accepted from either JSON body or form data as
    # "organization_id". Membership is verified -- you can't attach a
    # repo to an org you don't belong to, regardless of what id is sent.
    organization_id = None
    raw_org_id = (request.get_json(silent=True) or {}).get("organization_id") if request.is_json \
        else request.form.get("organization_id")
    if raw_org_id:
        if current_user.role_in_org(raw_org_id) is None:
            return jsonify({"error": "You are not a member of that organization"}), 403
        organization_id = int(raw_org_id)

    # Find or create a Repo row for this user+target (+organization), so
    # repeat scans of the same target build up history under one Repo
    # instead of creating duplicate entries every time. Personal and team
    # repos for the same target are intentionally kept separate rows.
    repo = Repo.query.filter_by(user_id=current_user.id, target=target, organization_id=organization_id).first()
    if not repo:
        display_name = target.rstrip("/").split("/")[-1] or target
        repo = Repo(user_id=current_user.id, target=target, name=display_name, organization_id=organization_id)
        if organization_id is not None:
            org = db.session.get(Organization, organization_id)
            repo.scan_frequency = org.default_scan_frequency
            if repo.scan_frequency != "off":
                repo.next_scheduled_at = datetime.now(timezone.utc) + _FREQUENCY_TO_TIMEDELTA[repo.scan_frequency]
        db.session.add(repo)
        db.session.commit()

    scan_id = _launch_scan(repo)
    return jsonify({"scan_id": scan_id})


@app.route("/repos")
@login_required
def repos_page():
    """Repo management / dashboard page -- lists every repo the user has
    saved, with its most recent scan status, and lets them re-scan or
    delete without re-entering the URL each time."""
    repos = get_user_accessible_repos(current_user)

    rows = ""
    for r in repos:
        last_scan = r.scans[0] if r.scans else None  # Repo.scans is already ordered desc
        if last_scan:
            last_when = f"{last_scan.created_at:%Y-%m-%d %H:%M}"
            if last_scan.triggered_by == "scheduled":
                last_when += ' <span style="color:#4a5268;">(auto)</span>'
            last_status = last_scan.status
            findings = (last_scan.code_findings_count or 0) + (last_scan.dep_findings_count or 0)
            findings_display = str(findings) if last_scan.status == "done" else "—"
        else:
            last_when, last_status, findings_display = "never", "—", "—"

        def _opt(value, label):
            selected = "selected" if r.scan_frequency == value else ""
            return f'<option value="{value}" {selected}>{label}</option>'

        schedule_form = f"""<form method="POST" action="/repos/{r.id}/schedule" style="display:inline;">
          <select name="frequency" class="btn-small" style="cursor:pointer;" onchange="this.form.requestSubmit()">
            {_opt("off", "Off")}
            {_opt("daily", "Daily")}
            {_opt("weekly", "Weekly")}
          </select>
        </form>"""

        scope_display = (
            f'<span style="color:#8890a0;">{r.organization.name}</span>'
            if r.organization_id else
            '<span style="color:#4a5268;">Personal</span>'
        )

        can_manage_access = (
            r.organization_id is not None
            and ORG_ROLE_RANK.get(current_user.role_in_org(r.organization_id), -1) >= ORG_ROLE_RANK["admin"]
        )
        access_link = f'<a href="/repos/{r.id}/access" class="btn-small" style="text-decoration:none;">Access</a>' \
            if can_manage_access else ""

        rows += f"""<tr>
          <td>{r.name}</td>
          <td class="target">{r.target}</td>
          <td>{scope_display}</td>
          <td>{last_when}</td>
          <td>{last_status}</td>
          <td>{findings_display}</td>
          <td>{schedule_form}</td>
          <td>
            <form method="POST" action="/repos/{r.id}/scan" style="display:inline;">
              <button type="submit" class="btn-small">Re-scan</button>
            </form>
            {access_link}
            <form method="POST" action="/repos/{r.id}/delete" style="display:inline;"
                  onsubmit="return confirm('Delete {r.name} and all its scan history?');">
              <button type="submit" class="btn-small btn-danger">Delete</button>
            </form>
          </td>
        </tr>"""

    return render_template_string(REPOS_UI, rows=rows)


@app.route("/repos/<int:repo_id>/access")
@login_required
def repo_access_page(repo_id):
    """Manage restriction + per-user access grants on a single org repo.
    Org owners/admins only -- plain members never see this page (there's
    nothing for them to manage, and it would leak who else has push
    rights unnecessarily)."""
    repo = db.session.get(Repo, repo_id)
    if repo is None or repo.is_personal() or not repo.can_be_accessed_by(current_user):
        return "Repo not found", 404

    role = current_user.role_in_org(repo.organization_id)
    if ORG_ROLE_RANK.get(role, -1) < ORG_ROLE_RANK["admin"]:
        return "Forbidden", 403

    org = repo.organization
    granted_user_ids = {g.user_id for g in repo.access_grants}

    grant_rows = ""
    for g in repo.access_grants:
        grant_rows += f"""<tr>
          <td>{g.user.email}</td>
          <td>{"yes" if g.can_push else "no"}</td>
          <td><button class="btn-small btn-danger"
                onclick="revokeAccess({repo.id}, {g.user_id}, '{g.user.email}')">Revoke</button></td>
        </tr>"""

    # Members not yet granted access, for the "add access" dropdown --
    # only relevant when the repo is restricted.
    ungranted = [m for m in org.memberships if m.user_id not in granted_user_ids]
    member_options = "".join(f'<option value="{m.user.email}">{m.user.email}</option>' for m in ungranted)

    return render_template_string(
        REPO_ACCESS_UI, repo=repo, org=org, grant_rows=grant_rows, member_options=member_options
    )


@app.route("/repos/<int:repo_id>/scan", methods=["POST"])
@login_required
def rescan_repo(repo_id):
    repo = get_accessible_repo_or_404(repo_id)
    if not repo:
        return "Repo not found", 404

    scan_id = _launch_scan(repo)
    return redirect(url_for("index") + f"?scan_id={scan_id}")


@app.route("/repos/<int:repo_id>/delete", methods=["POST"])
@login_required
def delete_repo(repo_id):
    repo = get_accessible_repo_or_404(repo_id)
    if not repo:
        return "Repo not found", 404

    db.session.delete(repo)  # cascades to Scan rows, per models.py relationship
    db.session.commit()
    return redirect(url_for("repos_page"))


@app.route("/approve/<scan_id>", methods=["POST"])
@login_required
def approve(scan_id):
    with SCANS_LOCK:
        scan = SCANS.get(scan_id)
        if not scan or not can_access_in_memory_scan(scan):
            return jsonify({"error": "Scan not found"}), 404
        if scan["status"] != "waiting_approval":
            return jsonify({"error": "Scan is not currently waiting for approval"}), 409

        data = request.get_json() or {}

        # Preferred: an explicit answer string, matching exactly what the
        # CLI accepts at the same checkpoint ("approve", "approve_all",
        # "reject") -- this is what lets the web UI offer the same
        # "confirmed fixes only" vs "include withheld fixes" choice the
        # CLI has always had. Falls back to the old boolean "approved"
        # field for backward compatibility with any existing caller that
        # only ever sends true/false.
        answer = data.get("answer")
        if answer not in ("approve", "approve_all", "reject"):
            answer = "approve" if data.get("approved") else "reject"

        # The PR checkpoint is where a real PR actually gets opened --
        # gate *approving* it on push rights specifically. Anyone who can
        # access the scan can still reject at this checkpoint (rejecting
        # doesn't open anything, there's nothing to protect), and can
        # still approve the earlier report checkpoint regardless of push
        # rights; this only tightens the final, consequential "yes, open
        # a real PR" action.
        if scan.get("checkpoint") == "pr_approval" and answer in ("approve", "approve_all"):
            repo_id = scan.get("repo_id")
            repo = db.session.get(Repo, repo_id) if repo_id is not None else None
            if repo is not None and not repo.can_be_pushed_by(current_user):
                return jsonify({"error": "You don't have permission to open a PR for this repo"}), 403

        scan["approval_answer"] = answer
        scan["approval_event"].set()

    return jsonify({"ok": True, "answer": answer})


@app.route("/stream/<scan_id>")
@login_required
def stream(scan_id):
    """Server-Sent Events endpoint — pushes events to the browser in real time."""
    scan = SCANS.get(scan_id)
    if not scan or not can_access_in_memory_scan(scan):
        return jsonify({"error": "Scan not found"}), 404

    def event_generator():
        last_index = 0
        while True:
            with SCANS_LOCK:
                events = SCANS[scan_id]["events"]
                status = SCANS[scan_id]["status"]
                new_events = events[last_index:]
                last_index = len(events)

            for ev in new_events:
                yield f"data: {json.dumps(ev)}\n\n"

            if status in ("done", "error") and not new_events:
                yield f"data: {json.dumps({'type': 'close'})}\n\n"
                break

            import time
            time.sleep(0.3)

    return Response(event_generator(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/scan/<scan_id>/fixes")
@login_required
def scan_fixes(scan_id):
    """JSON list of fixes for the diff-viewer modal's file list --
    NOT the diffs themselves (see /scan/<id>/diff/<idx> for that), just
    enough to render a clickable list: path, confirmed/withheld,
    critic score, finding count. Access-controlled the same way as
    /stream/<scan_id> above -- in-memory only (SCANS dict), no DB
    fallback, since fixes_for_diff (see run_scan_thread) is never
    persisted to the database, only kept in-process for the lifetime
    of the running app -- same lifetime tradeoff the existing /stream
    endpoint already accepts for live progress events.
    """
    with SCANS_LOCK:
        scan = SCANS.get(scan_id)
    if not scan or not can_access_in_memory_scan(scan):
        return jsonify({"error": "Scan not found"}), 404
    if scan["status"] != "done":
        return jsonify({"error": "Scan not finished yet"}), 202

    fixes = scan.get("fixes", [])
    out = []
    for i, f in enumerate(fixes):
        critique = f.get("critique", {}) or {}
        out.append({
            "index": i,
            "path": f.get("relative_path", "unknown"),
            "confirmed": f.get("confirmed", False),
            "critic_score": critique.get("score"),
            "critic_verdict": critique.get("verdict", "unavailable"),
            "findings_count": f.get("findings_addressed", 0),
        })
    return jsonify({"fixes": out})


@app.route("/scan/<scan_id>/diff/<int:idx>")
@login_required
def scan_diff(scan_id, idx):
    """Renders one fix's unified diff + verification checklist as an
    HTML fragment (not a full page -- this is fetched and injected
    into the diff-viewer modal's body via JS, same pattern the
    approval panel already uses for its dynamically-built content).
    """
    with SCANS_LOCK:
        scan = SCANS.get(scan_id)
    if not scan or not can_access_in_memory_scan(scan):
        return "Scan not found", 404
    if scan["status"] != "done":
        return "Scan not finished yet", 202

    fixes = scan.get("fixes", [])
    if idx < 0 or idx >= len(fixes):
        return "Fix not found", 404

    fix = fixes[idx]
    original_code = fix.get("original_code", "")
    fixed_code = fix.get("fixed_code", "")
    path = fix.get("relative_path", "unknown")
    confirmed = fix.get("confirmed", False)

    diff_html = _render_unified_diff_html(original_code, fixed_code)
    checklist_html = _render_verification_checklist_html(fix)
    status_label = "CONFIRMED" if confirmed else "WITHHELD -- NOT MERGED"
    status_class = "check-pass" if confirmed else "check-fail"

    return f"""
    <div class="diff-file-header">
      <span class="diff-file-path">{html_lib.escape(path)}</span>
      <span class="diff-check-badge {status_class}">{status_label}</span>
    </div>
    <div class="diff-section-label">Verification</div>
    <div class="diff-checklist">{checklist_html}</div>
    <div class="diff-section-label">Diff</div>
    <div class="diff-view">{diff_html}</div>
    """



@app.route("/scan/<scan_id>/sbom.json")
@login_required
def serve_sbom(scan_id):
    """Serve the CycloneDX SBOM file, same auth/fallback pattern as
    serve_report. VULNERABILITY-ONLY scope -- see
    tools/sbom_generator.py's module docstring; the file itself also
    carries a "vuln-agent:sbom-scope" property saying the same thing,
    so the caveat travels with the file even if downloaded elsewhere."""
    with SCANS_LOCK:
        scan = SCANS.get(scan_id)

    if scan and can_access_in_memory_scan(scan):
        if scan["status"] != "done":
            return "SBOM not ready yet", 202
        sbom_path = scan.get("sbom_path")
    else:
        scan_row = db.session.get(Scan, scan_id)
        if not scan_row or not can_access_scan_row(scan_row):
            return "Scan not found", 404
        if scan_row.status != "done":
            return "SBOM not ready yet", 202
        sbom_path = scan_row.sbom_path

    if not sbom_path or not os.path.exists(sbom_path):
        return "SBOM file not found (scan may predate SBOM support)", 404

    return send_file(sbom_path, mimetype="application/json", as_attachment=True, download_name=f"sbom_{scan_id}.json")


@app.route("/report/<scan_id>")
@login_required
def serve_report(scan_id):
    """Serve the finished HTML report.

    Falls back to the database if the scan isn't in the in-memory SCANS
    dict (e.g. the process restarted since the scan ran) -- report files
    on disk outlive the process, so history should too.
    """
    with SCANS_LOCK:
        scan = SCANS.get(scan_id)

    if scan and can_access_in_memory_scan(scan):
        if scan["status"] != "done":
            return "Report not ready yet", 202
        report_path = scan.get("report_path")
    else:
        scan_row = db.session.get(Scan, scan_id)
        if not scan_row or not can_access_scan_row(scan_row):
            return "Scan not found", 404
        if scan_row.status != "done":
            return "Report not ready yet", 202
        report_path = scan_row.report_path

    if not report_path or not os.path.exists(report_path):
        return "Report file not found", 404

    return send_file(report_path)


@app.route("/history")
@login_required
def history():
    """Simple scan history page -- every past scan the user can access:
    their own personal scans, PLUS every scan run against any repo
    belonging to an organization they're a member of. Newest first."""
    accessible_repo_ids = [r.id for r in get_user_accessible_repos(current_user)]
    scans = (
        Scan.query.filter(Scan.repo_id.in_(accessible_repo_ids) if accessible_repo_ids else db.false())
        .order_by(Scan.created_at.desc())
        .all()
    )

    rows = ""
    for s in scans:
        repo_name = s.repo.name if s.repo else "unknown"
        pr_link = f'<a href="{s.pr_url}" target="_blank">PR</a>' if s.pr_url else "—"
        report_link = f'<a href="/report/{s.id}">report</a>' if s.status == "done" else "—"
        sbom_link = f'<a href="/scan/{s.id}/sbom.json">sbom</a>' if s.status == "done" and s.sbom_path else "—"
        findings = (s.code_findings_count or 0) + (s.dep_findings_count or 0)
        trigger_label = "auto" if s.triggered_by == "scheduled" else "manual"
        rows += (
            f"<tr><td>{repo_name}</td><td>{s.created_at:%Y-%m-%d %H:%M}</td>"
            f"<td>{s.status}</td><td>{trigger_label}</td><td>{findings}</td>"
            f"<td>{s.code_fixes_count or 0}</td><td>{s.withheld_fixes_count or 0}</td>"
            f"<td>{report_link}</td><td>{sbom_link}</td><td>{pr_link}</td></tr>"
        )

    return render_template_string(HISTORY_UI, rows=rows)


@app.route("/admin/scans")
@login_required
@require_role("admin")
def admin_all_scans():
    """Admin-only: every scan across every user, not just current_user's
    own. Distinct from /history (which is always scoped to the logged-in
    user regardless of role) -- this is the one place role actually
    changes what data is visible, so it's a deliberate, narrow exception
    to the "always filter by user_id" rule everywhere else in this file.
    """
    scans = Scan.query.order_by(Scan.created_at.desc()).limit(500).all()

    results = []
    for s in scans:
        results.append({
            "scan_id": s.id,
            "user_email": s.owner.email if s.owner else "unknown",
            "repo_name": s.repo.name if s.repo else "unknown",
            "status": s.status,
            "triggered_by": s.triggered_by,
            "created_at": s.created_at.isoformat(),
            "code_findings": s.code_findings_count,
            "dep_findings": s.dep_findings_count,
            "pr_url": s.pr_url,
        })

    return jsonify({"total": len(results), "scans": results})


@app.route("/debug-sentry")
def debug_sentry():
    """Deliberately throws an unhandled exception -- exists purely to
    confirm Sentry is wired up correctly (visit this once, check the
    Sentry Issues feed, then feel free to delete this route once
    confirmed). Division by zero is Sentry's own standard example for
    exactly this reason: it's unambiguous and can't be caused by bad
    input, so seeing it in Sentry proves the integration works."""
    return 1 / 0


@app.route("/metrics")
def prometheus_metrics():
    """Scrape target for a real Prometheus server (see docker-compose.yml).
    Deliberately NOT behind @login_required -- Prometheus's scraper has
    no session/cookie to authenticate with, and this endpoint exposes
    only aggregate counters (no per-user or per-repo data), matching
    the usual convention for /metrics endpoints. If this needs to be
    locked down in a real deployment, restrict at the network/reverse-
    proxy level instead (e.g. only allow the Prometheus container's IP)
    rather than adding app-level auth that Prometheus can't satisfy.
    """
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/admin/metrics")
@login_required
@require_role("admin")
def admin_metrics():
    """Admin-only observability dashboard. Aggregates directly from the
    Scan table -- no new tables needed, since created_at/completed_at
    and the *_count columns were already being populated per scan.

    NOTE: this only gives whole-scan duration (completed_at - created_at),
    not a per-pipeline-stage breakdown (code_scan vs fix_generate vs...).
    Getting that would mean parsing per-node timestamps out of the log
    file, or adding stage timestamps to the Scan model -- a reasonable
    follow-up once this baseline version is in place.

    Gated by the same @require_role("admin") decorator as /admin/scans --
    global platform-admin flag only (User.role == "admin"), NOT
    per-organization owner/admin. An org owner/admin who isn't also a
    global admin cannot see this page, same as they can't see /admin/scans.
    """
    now = datetime.now(timezone.utc)
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)

    total_scans = db.session.query(func.count(Scan.id)).scalar() or 0

    status_counts = dict(
        db.session.query(Scan.status, func.count(Scan.id))
        .group_by(Scan.status)
        .all()
    )

    scans_24h = db.session.query(func.count(Scan.id)).filter(Scan.created_at >= last_24h).scalar() or 0
    scans_7d = db.session.query(func.count(Scan.id)).filter(Scan.created_at >= last_7d).scalar() or 0

    # Average duration, completed scans only (completed_at IS NOT NULL).
    # extract(epoch from ...) gives seconds directly in Postgres.
    avg_duration_seconds = db.session.query(
        func.avg(func.extract("epoch", Scan.completed_at - Scan.created_at))
    ).filter(Scan.completed_at.isnot(None)).scalar()

    error_count = status_counts.get("error", 0)
    error_rate = round((error_count / total_scans) * 100, 1) if total_scans else 0.0

    fix_totals = db.session.query(
        func.coalesce(func.sum(Scan.code_fixes_count), 0),
        func.coalesce(func.sum(Scan.withheld_fixes_count), 0),
        func.coalesce(func.sum(Scan.code_findings_count), 0),
        func.coalesce(func.sum(Scan.dep_findings_count), 0),
    ).one()
    confirmed_fixes, withheld_fixes, code_findings_total, dep_findings_total = fix_totals

    total_fix_attempts = confirmed_fixes + withheld_fixes
    confirm_rate = round((confirmed_fixes / total_fix_attempts) * 100, 1) if total_fix_attempts else None

    payload = {
        "generated_at": now.isoformat(),
        "totals": {
            "all_scans": total_scans,
            "last_24h": scans_24h,
            "last_7d": scans_7d,
        },
        "by_status": {
            "running": status_counts.get("running", 0),
            "waiting_approval": status_counts.get("waiting_approval", 0),
            "done": status_counts.get("done", 0),
            "error": status_counts.get("error", 0),
        },
        "performance": {
            "avg_duration_seconds": round(avg_duration_seconds, 1) if avg_duration_seconds is not None else None,
            "error_rate_pct": error_rate,
        },
        "fix_pipeline": {
            "confirmed_fixes": confirmed_fixes,
            "withheld_fixes": withheld_fixes,
            "confirm_rate_pct": confirm_rate,
        },
        "findings": {
            "code_findings_total": code_findings_total,
            "dep_findings_total": dep_findings_total,
        },
    }

    if request.args.get("format") == "json":
        return jsonify(payload)

    return render_template_string(METRICS_UI, m=payload)


@app.route("/status/<scan_id>")
@login_required
def scan_status(scan_id):
    with SCANS_LOCK:
        scan = SCANS.get(scan_id)
    if not scan or not can_access_in_memory_scan(scan):
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "status": scan["status"],
        "stats": scan.get("stats"),
        "error": scan.get("error")
    })


# ---------------------------------------------------------------------------
# Organizations / teams -- create, list, and manage per-org membership+roles.
# Per-org roles (owner/admin/member) are entirely separate from the global
# User.role platform-admin flag checked by require_role() above -- see the
# module docstring in models.py for the full breakdown.
# ---------------------------------------------------------------------------
@app.route("/organizations", methods=["GET"])
@login_required
def list_organizations():
    """Every organization the current user belongs to, with their role
    in each -- a user with zero organizations just gets an empty list;
    they can keep using personal repos exactly as before."""
    results = []
    for m in current_user.memberships:
        org = m.organization
        results.append({
            "id": org.id, "name": org.name, "role": m.role,
            "member_count": len(org.memberships),
        })
    return jsonify({"organizations": results})


@app.route("/organizations", methods=["POST"])
@login_required
def create_organization():
    """Creates a new organization. The creator is automatically its
    first member with role 'owner' -- every org must have at least one
    owner, and whoever starts it is the natural first one."""
    data = request.get_json(silent=True) or request.form
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Organization name is required"}), 400

    org = Organization(name=name)
    db.session.add(org)
    db.session.flush()  # assigns org.id before we reference it below

    membership = Membership(user_id=current_user.id, organization_id=org.id, role="owner")
    db.session.add(membership)
    db.session.commit()

    return jsonify({"id": org.id, "name": org.name, "role": "owner"}), 201


@app.route("/organizations/<int:org_id>/members", methods=["GET"])
@login_required
def list_org_members(org_id):
    """Any member (any role) can see who else is in their org."""
    if current_user.role_in_org(org_id) is None:
        return jsonify({"error": "Organization not found"}), 404

    org = db.session.get(Organization, org_id)
    members = [{"user_id": m.user_id, "email": m.user.email, "role": m.role,
                "joined_at": m.joined_at.isoformat()} for m in org.memberships]
    return jsonify({"organization": org.name, "members": members})


@app.route("/organizations/<int:org_id>/members", methods=["POST"])
@login_required
@require_org_role("admin")
def add_org_member(org_id):
    """Adds an existing user (by email) to the org. Requires 'admin' or
    'owner' in THIS org -- being an admin of a different org doesn't
    count, per-org roles don't carry across organizations.
    New members can only be added as 'member' or 'admin' here -- setting
    someone straight to 'owner' goes through change_org_member_role
    below, which itself requires being an owner (not just admin)."""
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    role = data.get("role", "member")

    if role not in ("member", "admin"):
        return jsonify({"error": "New members must be added as 'member' or 'admin'"}), 400

    user = User.query.filter(db.func.lower(User.email) == email).first()

    if current_user.role_in_org(org_id) != "owner" and role == "admin":
        return jsonify({"error": "Only an owner can add a member as admin"}), 403

    if not user:
        # No account with this email yet -- create a pending invite that
        # gets turned into a real Membership automatically the moment
        # someone signs up with this address (see signup()).
        if OrgInvite.query.filter_by(organization_id=org_id, email=email).first():
            return jsonify({"error": "An invite is already pending for that email"}), 409
        invite = OrgInvite(organization_id=org_id, email=email, role=role, invited_by_user_id=current_user.id)
        db.session.add(invite)
        db.session.commit()
        return jsonify({"email": email, "role": role, "pending": True}), 201

    if Membership.query.filter_by(user_id=user.id, organization_id=org_id).first():
        return jsonify({"error": "That user is already a member of this organization"}), 409

    membership = Membership(user_id=user.id, organization_id=org_id, role=role)
    db.session.add(membership)
    db.session.commit()

    return jsonify({"user_id": user.id, "email": user.email, "role": role}), 201


@app.route("/organizations/<int:org_id>/invites", methods=["GET"])
@login_required
def list_org_invites(org_id):
    """Pending (not-yet-accepted) invites for this org. Any member can
    see the list, same visibility rule as list_org_members."""
    if current_user.role_in_org(org_id) is None:
        return jsonify({"error": "Organization not found"}), 404

    invites = OrgInvite.query.filter_by(organization_id=org_id).all()
    return jsonify({"invites": [
        {"id": i.id, "email": i.email, "role": i.role, "created_at": i.created_at.isoformat()}
        for i in invites
    ]})


@app.route("/organizations/<int:org_id>/invites/<int:invite_id>", methods=["DELETE"])
@login_required
@require_org_role("admin")
def revoke_org_invite(org_id, invite_id):
    """Cancel a pending invite before it's accepted."""
    invite = OrgInvite.query.filter_by(id=invite_id, organization_id=org_id).first()
    if invite is None:
        return jsonify({"error": "Invite not found"}), 404
    db.session.delete(invite)
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/organizations/<int:org_id>/members/<int:user_id>/role", methods=["POST"])
@login_required
@require_org_role("owner")
def change_org_member_role(org_id, user_id):
    """Changing anyone's role -- including promoting to/demoting from
    'owner' -- requires being an owner yourself. An admin can add plain
    members (see add_org_member) but cannot grant ownership."""
    data = request.get_json(silent=True) or request.form
    new_role = data.get("role")
    if new_role not in ("member", "admin", "owner"):
        return jsonify({"error": "role must be one of: member, admin, owner"}), 400

    membership = Membership.query.filter_by(user_id=user_id, organization_id=org_id).first()
    if not membership:
        return jsonify({"error": "That user is not a member of this organization"}), 404

    if membership.role == "owner" and new_role != "owner":
        remaining_owners = Membership.query.filter_by(organization_id=org_id, role="owner").count()
        if remaining_owners <= 1:
            return jsonify({"error": "Cannot demote the last remaining owner of an organization"}), 409

    membership.role = new_role
    db.session.commit()
    return jsonify({"user_id": user_id, "role": new_role})


@app.route("/organizations/<int:org_id>/members/<int:user_id>", methods=["DELETE"])
@login_required
@require_org_role("admin")
def remove_org_member(org_id, user_id):
    """Removing a member requires 'admin' or 'owner'. An admin cannot
    remove an owner (only another owner can) -- and the last owner can
    never be removed, same protection as change_org_member_role above."""
    membership = Membership.query.filter_by(user_id=user_id, organization_id=org_id).first()
    if not membership:
        return jsonify({"error": "That user is not a member of this organization"}), 404

    if membership.role == "owner":
        if current_user.role_in_org(org_id) != "owner":
            return jsonify({"error": "Only an owner can remove another owner"}), 403
        remaining_owners = Membership.query.filter_by(organization_id=org_id, role="owner").count()
        if remaining_owners <= 1:
            return jsonify({"error": "Cannot remove the last remaining owner of an organization"}), 409

    db.session.delete(membership)
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/organizations/<int:org_id>/settings", methods=["POST"])
@login_required
@require_org_role("admin")
def update_org_settings(org_id):
    """Org-level defaults. Currently just default_scan_frequency, applied
    to new repos added to this org going forward -- doesn't retroactively
    change existing repos' schedules."""
    data = request.get_json(silent=True) or request.form
    frequency = data.get("default_scan_frequency")
    if frequency not in ("off", "daily", "weekly"):
        return jsonify({"error": "default_scan_frequency must be 'off', 'daily', or 'weekly'"}), 400

    org = db.session.get(Organization, org_id)
    org.default_scan_frequency = frequency
    db.session.commit()
    return jsonify({"default_scan_frequency": org.default_scan_frequency})


@app.route("/repos/<int:repo_id>/access/grants", methods=["GET"])
@login_required
def list_repo_access(repo_id):
    """Current restriction flag + explicit access grants for an org repo,
    as JSON (used by tests/API consumers -- the HTML management page at
    GET /repos/<id>/access renders the same data server-side directly,
    so it doesn't call this route itself). Requires being able to see
    the repo at all (org owners/admins always can; restricted members
    only if already granted -- which is exactly the visibility rule we
    want here too)."""
    repo = get_accessible_repo_or_404(repo_id)
    if repo is None or repo.is_personal():
        return jsonify({"error": "Repo not found"}), 404

    grants = [
        {"user_id": g.user_id, "email": g.user.email, "can_push": g.can_push}
        for g in repo.access_grants
    ]
    return jsonify({"restricted": repo.restricted, "grants": grants})


@app.route("/repos/<int:repo_id>/access", methods=["POST"])
@login_required
def update_repo_access(repo_id):
    """Toggle repo.restricted and/or upsert a single user's access grant.
    Only org owners/admins may manage this -- same rank required to
    manage membership, since restricting a repo is itself a membership-
    adjacent decision about who on the team can reach it."""
    repo = db.session.get(Repo, repo_id)
    if repo is None or repo.is_personal() or repo.can_be_accessed_by(current_user) is False:
        return jsonify({"error": "Repo not found"}), 404

    role = current_user.role_in_org(repo.organization_id)
    if ORG_ROLE_RANK.get(role, -1) < ORG_ROLE_RANK["admin"]:
        return jsonify({"error": "Only an org admin or owner can manage repo access"}), 403

    data = request.get_json(silent=True) or request.form

    if "restricted" in data:
        repo.restricted = bool(data.get("restricted"))

    grant_email = (data.get("email") or "").strip().lower()
    if grant_email:
        user = User.query.filter(db.func.lower(User.email) == grant_email).first()
        if not user:
            return jsonify({"error": "No user found with that email"}), 404
        if current_user.role_in_org(repo.organization_id) is None or user.role_in_org(repo.organization_id) is None:
            return jsonify({"error": "That user is not a member of this organization"}), 400

        grant = RepoAccess.query.filter_by(repo_id=repo.id, user_id=user.id).first()
        can_push = bool(data.get("can_push", False))
        if grant is None:
            grant = RepoAccess(repo_id=repo.id, user_id=user.id, can_push=can_push,
                                granted_by_user_id=current_user.id)
            db.session.add(grant)
        else:
            grant.can_push = can_push

    db.session.commit()
    return jsonify({"restricted": repo.restricted})


@app.route("/repos/<int:repo_id>/access/<int:user_id>", methods=["DELETE"])
@login_required
def revoke_repo_access(repo_id, user_id):
    """Remove a user's explicit access grant on a restricted repo."""
    repo = db.session.get(Repo, repo_id)
    if repo is None or repo.is_personal():
        return jsonify({"error": "Repo not found"}), 404

    role = current_user.role_in_org(repo.organization_id)
    if ORG_ROLE_RANK.get(role, -1) < ORG_ROLE_RANK["admin"]:
        return jsonify({"error": "Only an org admin or owner can manage repo access"}), 403

    grant = RepoAccess.query.filter_by(repo_id=repo_id, user_id=user_id).first()
    if grant is None:
        return jsonify({"error": "No access grant found for that user"}), 404
    db.session.delete(grant)
    db.session.commit()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Organizations UI (HTML pages) -- thin wrappers around the JSON API above.
# Kept as separate routes (/orgs, /orgs/<id>) rather than content-negotiating
# on /organizations, so the existing JSON API (already verified end-to-end)
# is untouched.
# ---------------------------------------------------------------------------

@app.route("/orgs")
@login_required
def orgs_page():
    """List every organization the current user belongs to, plus a form
    to create a new one. Mirrors the server-rendered-rows pattern used by
    /repos and /history -- current_user.memberships is already loaded,
    no extra JSON round-trip needed for the initial view."""
    rows = ""
    for m in current_user.memberships:
        org = m.organization
        rows += f"""<tr>
          <td>{org.name}</td>
          <td>{m.role}</td>
          <td>{len(org.memberships)}</td>
          <td><a href="/orgs/{org.id}">manage</a></td>
        </tr>"""

    return render_template_string(ORGS_UI, rows=rows)


@app.route("/orgs/<int:org_id>")
@login_required
def org_detail_page(org_id):
    """Single-org management page: member list + role controls + invite
    form. 404s if the current user isn't a member at all -- same rule
    the JSON API uses, so a non-member can't confirm the org exists."""
    my_role = current_user.role_in_org(org_id)
    if my_role is None:
        return "Organization not found", 404

    org = db.session.get(Organization, org_id)
    can_manage = ORG_ROLE_RANK.get(my_role, -1) >= ORG_ROLE_RANK["admin"]
    is_owner = my_role == "owner"

    member_rows = ""
    for m in org.memberships:
        role_cell = m.role
        if is_owner:
            role_cell = f"""<select class="btn-small" style="cursor:pointer;"
                onchange="changeRole({org.id}, {m.user_id}, this.value)">
                <option value="member" {"selected" if m.role == "member" else ""}>member</option>
                <option value="admin" {"selected" if m.role == "admin" else ""}>admin</option>
                <option value="owner" {"selected" if m.role == "owner" else ""}>owner</option>
              </select>"""

        remove_cell = ""
        if can_manage and not (m.role == "owner" and not is_owner):
            remove_cell = f"""<button class="btn-small btn-danger"
                onclick="removeMember({org.id}, {m.user_id}, '{m.user.email}')">Remove</button>"""

        member_rows += f"""<tr>
          <td>{m.user.email}</td>
          <td>{role_cell}</td>
          <td>{m.joined_at:%Y-%m-%d}</td>
          <td>{remove_cell}</td>
        </tr>"""

    invite_form = ""
    if can_manage:
        role_choices = '<option value="member">member</option>'
        if is_owner:
            role_choices += '<option value="admin">admin</option>'
        invite_form = f"""
        <div class="invite-box">
          <h2>Invite a member</h2>
          <div class="invite-row">
            <input type="email" id="invite-email" placeholder="teammate@example.com">
            <select id="invite-role">{role_choices}</select>
            <button class="btn-small" onclick="inviteMember({org.id})">Invite</button>
          </div>
          <div class="invite-msg" id="invite-msg"></div>
        </div>"""

    invite_rows = ""
    if can_manage:
        for inv in org.invites:
            invite_rows += f"""<tr>
              <td>{inv.email}</td>
              <td>{inv.role} <span style="color:#4a5268;">(pending)</span></td>
              <td>{inv.created_at:%Y-%m-%d}</td>
              <td><button class="btn-small btn-danger"
                    onclick="revokeInvite({org.id}, {inv.id}, '{inv.email}')">Revoke</button></td>
            </tr>"""

    settings_box = ""
    if can_manage:
        freq_opts = "".join(
            f'<option value="{f}" {"selected" if org.default_scan_frequency == f else ""}>{f}</option>'
            for f in ("off", "daily", "weekly")
        )
        settings_box = f"""
        <div class="invite-box">
          <h2>Org settings</h2>
          <label style="font-size:0.8rem;color:#8890a0;">Default scan frequency for new repos</label>
          <div class="invite-row" style="margin-top:0.4rem;">
            <select id="default-freq">{freq_opts}</select>
            <button class="btn-small" onclick="saveSettings({org.id})">Save</button>
          </div>
          <div class="invite-msg" id="settings-msg"></div>
        </div>"""

    return render_template_string(
        ORG_DETAIL_UI, org=org, my_role=my_role, member_rows=member_rows,
        invite_form=invite_form, invite_rows=invite_rows, settings_box=settings_box,
    )


# ---------------------------------------------------------------------------
# Auth pages (login / signup) -- minimal, matches the dark theme used by
# the main scan UI so it doesn't feel bolted-on.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Bugatti design system — shared CSS (austere luxury: black canvas, white
# letterspaced display, serif body, monospace labels, weight 400, transparent
# pill buttons). Concatenated into templates (NOT an f-string) so Jinja braces
# in the surrounding markup stay intact.
# ---------------------------------------------------------------------------
BUGATTI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@400&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --primary:#fff;--ink:#fff;--body:#ccc;--body-strong:#e6e6e6;--muted:#999;--muted-soft:#666;
  --hairline:#262626;--hairline-strong:#3a3a3a;--canvas:#000;--surface-soft:#0d0d0d;
  --surface-card:#141414;--surface-elevated:#1f1f1f;--on-primary:#000;--on-dark:#fff;
  --link:#c3d9f3;--warning:#d4a017;--success:#5fa657;
  --font-display:'Saira Condensed',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  --font-text:'Cormorant Garamond',Garamond,'Times New Roman',serif;
  --font-mono:'JetBrains Mono',ui-monospace,'SF Mono','Cascadia Mono',monospace;
  --xxs:4px;--xs:8px;--sm:12px;--md:16px;--lg:24px;--xl:40px;--xxl:64px;--section:120px;
  --r-none:0px;--r-pill:9999px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body{background:var(--canvas);color:var(--on-dark);font-family:var(--font-text);
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}
.display-xl{font-family:var(--font-display);font-weight:400;font-size:64px;line-height:1.1;letter-spacing:4px;text-transform:uppercase;}
.display-lg{font-family:var(--font-display);font-weight:400;font-size:48px;line-height:1.15;letter-spacing:3px;text-transform:uppercase;}
.display-md{font-family:var(--font-display);font-weight:400;font-size:32px;line-height:1.2;letter-spacing:2px;text-transform:uppercase;}
.display-sm{font-family:var(--font-display);font-weight:400;font-size:24px;line-height:1.3;letter-spacing:1.5px;text-transform:uppercase;}
.title-md{font-family:var(--font-display);font-weight:400;font-size:20px;line-height:1.3;letter-spacing:1px;text-transform:uppercase;}
.title-sm{font-family:var(--font-display);font-weight:400;font-size:16px;line-height:1.3;letter-spacing:1.5px;text-transform:uppercase;}
.caption{font-family:var(--font-mono);font-weight:400;font-size:11px;line-height:1.4;letter-spacing:2px;text-transform:uppercase;color:var(--muted);}
.body-md{font-family:var(--font-text);font-weight:400;font-size:18px;line-height:1.5;color:var(--body);}
.body-sm{font-family:var(--font-text);font-weight:400;font-size:15px;line-height:1.5;color:var(--body);}
.nav-link{font-family:var(--font-mono);font-weight:400;font-size:12px;line-height:1.4;letter-spacing:2px;text-transform:uppercase;}
.wordmark{font-family:var(--font-display);font-weight:400;font-size:14px;letter-spacing:6px;text-transform:uppercase;color:var(--on-dark);text-decoration:none;}
.top-nav{height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 var(--xl);background:transparent;position:relative;}
.top-nav .center{position:absolute;left:50%;transform:translateX(-50%);}
.top-nav a{color:var(--on-dark);text-decoration:none;}
.top-nav .nav-group{display:flex;gap:var(--lg);align-items:center;}
.btn{display:inline-flex;align-items:center;justify-content:center;height:44px;padding:14px 32px;
  background:transparent;color:var(--on-dark);border:1px solid var(--on-dark);border-radius:var(--r-pill);
  font-family:var(--font-mono);font-size:14px;letter-spacing:2.5px;text-transform:uppercase;
  text-decoration:none;cursor:pointer;transition:background .25s ease,color .25s ease;}
.btn:hover{background:var(--on-dark);color:var(--on-primary);}
.btn-full{width:100%;}
.text-link{color:var(--link);font-family:var(--font-text);text-decoration:underline;text-underline-offset:3px;}
.mono-link{color:var(--muted);font-family:var(--font-mono);font-size:11px;letter-spacing:2px;text-transform:uppercase;text-decoration:none;}
.mono-link:hover{color:var(--on-dark);}
.field{display:flex;flex-direction:column;gap:var(--xs);}
.field label{font-family:var(--font-mono);font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);}
.input{height:44px;padding:12px 0;background:transparent;color:var(--on-dark);border:none;
  border-bottom:1px solid var(--hairline-strong);font-family:var(--font-text);font-size:18px;outline:none;
  transition:border-color .2s ease;width:100%;}
.input::placeholder{color:var(--muted);}
.input:focus{border-bottom-color:var(--on-dark);}
.card{background:var(--surface-card);border-radius:var(--r-none);padding:var(--lg);}
.footer{background:var(--canvas);color:var(--muted);padding:var(--xxl) var(--xl);border-top:1px solid var(--hairline);}
.container{max-width:1280px;margin:0 auto;padding:0 var(--xl);}
.stack-lg>*+*{margin-top:var(--lg);}
.center-text{text-align:center;}
.muted{color:var(--muted);}
"""


# ---------------------------------------------------------------------------
# Public marketing landing (editorial grid) — served at "/" for logged-out
# visitors. Logged-in users are redirected to /dashboard by the route.
# ---------------------------------------------------------------------------
LANDING_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sentinel — Autonomous vulnerability intelligence</title>
  <style>""" + BUGATTI_CSS + """
  .hero-band{position:relative;min-height:78vh;display:flex;flex-direction:column;align-items:center;
    justify-content:center;text-align:center;background:radial-gradient(140% 100% at 50% 0%,#1c1c1c 0%,#000 55%);
    border-bottom:1px solid var(--hairline);padding:0 var(--xl);}
  .hero-band h1{max-width:1000px;margin:var(--md) 0 var(--lg);}
  .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:var(--xl);padding:var(--section) var(--xl);}
  .model-card .thumb{aspect-ratio:16/9;margin-bottom:var(--md);background:linear-gradient(135deg,#161616,#000);border:1px solid var(--hairline);}
  .model-card .display-sm{margin-bottom:var(--xs);}
  .cta-band{position:relative;padding:var(--section) var(--xl);text-align:center;
    background:radial-gradient(100% 120% at 50% 100%,#1a1a1a,#000 60%);border-top:1px solid var(--hairline);}
  .cta-band h2{margin-bottom:var(--xl);}
  @media(max-width:768px){.display-xl{font-size:34px;}.grid-3{grid-template-columns:1fr;padding:var(--xxl) var(--xl);}}
  </style>
</head>
<body>
  <nav class="top-nav">
    <div class="nav-group"><span class="nav-link muted">Autonomous security</span></div>
    <a class="wordmark center" href="/">Sentinel</a>
    <div class="nav-group"><a class="nav-link" href="/login">Sign in</a></div>
  </nav>

  <header class="hero-band">
    <p class="caption">Continuous · Autonomous · Silent</p>
    <h1 class="display-xl">The perimeter,<br>rewritten</h1>
    <p class="body-md" style="max-width:480px;">A security agent that reads your code the way an attacker would — then closes the door first.</p>
    <div style="margin-top:var(--xl);display:flex;gap:var(--md);">
      <a class="btn" href="/signup">Request access</a>
      <a class="btn" href="/login" style="border-color:var(--hairline-strong);color:var(--muted);">Sign in</a>
    </div>
  </header>

  <section class="grid-3">
    <article class="model-card">
      <div class="thumb"></div>
      <p class="caption">Code · SAST</p>
      <h3 class="display-sm">Source analysis</h3>
      <p class="body-sm">Taint tracking across your own code, tuned for the languages you actually ship.</p>
    </article>
    <article class="model-card">
      <div class="thumb"></div>
      <p class="caption">Dependencies · SCA</p>
      <h3 class="display-sm">Supply chain</h3>
      <p class="body-sm">Full transitive graphs, enriched with live CVE and exploit-maturity signals.</p>
    </article>
    <article class="model-card">
      <div class="thumb"></div>
      <p class="caption">Remediation · PR</p>
      <h3 class="display-sm">Autonomous fixes</h3>
      <p class="body-sm">The patch, the test, the pull request — generated, reviewed, and ready to merge.</p>
    </article>
  </section>

  <section class="cta-band">
    <p class="caption">Ready when you are</p>
    <h2 class="display-lg">Discover Sentinel</h2>
    <a class="btn" href="/signup">Request access</a>
  </section>

  <footer class="footer center-text">
    <span class="wordmark">Sentinel</span>
    <p class="body-sm muted" style="margin-top:var(--md);">© 2026 Sentinel Security. All rights reserved.</p>
  </footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Authenticated overview dashboard (app-a). Stats + accessible-repo rows.
# Vars: user_email, repo_count, open_findings, done_count, last_scan, rows.
# ---------------------------------------------------------------------------
DASHBOARD_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sentinel — Overview</title>
  <style>""" + BUGATTI_CSS + """
  .shell{display:grid;grid-template-columns:240px 1fr;min-height:100vh;}
  .sidebar{border-right:1px solid var(--hairline);padding:var(--lg) 0;display:flex;flex-direction:column;position:sticky;top:0;height:100vh;}
  .sidebar .brand{padding:var(--md) var(--lg) var(--xl);}
  .side-nav{display:flex;flex-direction:column;}
  .side-nav a{padding:14px var(--lg);font-family:var(--font-mono);font-size:12px;letter-spacing:2px;
    text-transform:uppercase;color:var(--muted);text-decoration:none;border-left:2px solid transparent;}
  .side-nav a.active{color:var(--on-dark);border-left-color:var(--on-dark);}
  .side-nav a:hover{color:var(--on-dark);}
  .side-foot{margin-top:auto;padding:var(--lg);border-top:1px solid var(--hairline);}
  .main{padding:var(--xl);}
  .page-head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:var(--xl);}
  .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--hairline);margin-bottom:var(--xxl);}
  .stat{padding:var(--lg);border-right:1px solid var(--hairline);}
  .stat:last-child{border-right:none;}
  .stat .display-md{margin:var(--xs) 0;}
  .stat .warn{color:var(--warning);}.stat .ok{color:var(--success);}
  .list-head,.repo-row{display:grid;grid-template-columns:2fr 1fr 1fr 120px;gap:var(--lg);align-items:center;
    padding:var(--lg) 0;border-bottom:1px solid var(--hairline);}
  .list-head{border-bottom-color:var(--hairline-strong);}
  .list-head span{font-family:var(--font-mono);font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--muted-soft);}
  .repo-row .name{font-family:var(--font-display);font-size:18px;letter-spacing:1px;text-transform:uppercase;}
  .repo-row .sub{font-family:var(--font-mono);font-size:11px;letter-spacing:1px;color:var(--muted);}
  .sev{font-family:var(--font-mono);font-size:11px;letter-spacing:1px;text-transform:uppercase;}
  .sev.high{color:var(--warning);}.sev.clear{color:var(--success);}
  .empty{padding:var(--xxl) 0;text-align:center;}
  @media(max-width:900px){.shell{grid-template-columns:1fr;}
    .sidebar{position:static;height:auto;flex-direction:row;flex-wrap:wrap;align-items:center;}
    .side-foot{display:none;}.stat-grid{grid-template-columns:1fr 1fr;}
    .list-head{display:none;}.repo-row{grid-template-columns:1fr 1fr;}}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <a class="wordmark brand" href="/">Sentinel</a>
      <nav class="side-nav">
        <a href="/dashboard" class="active">Overview</a>
        <a href="/repos">Repositories</a>
        <a href="/history">History</a>
        <a href="/scan-console">New scan</a>
        <a href="/account">Account</a>
        <a href="/logout">Log out</a>
      </nav>
      <div class="side-foot">
        <p class="caption">Signed in as</p>
        <p class="body-sm" style="color:var(--on-dark);">{{ user_email }}</p>
      </div>
    </aside>

    <main class="main">
      <div class="page-head">
        <div>
          <p class="caption">Workspace</p>
          <h1 class="display-lg">Overview</h1>
        </div>
        <a class="btn" href="/scan-console">New scan</a>
      </div>

      <section class="stat-grid">
        <div class="stat"><p class="caption">Repositories</p><div class="display-md">{{ repo_count }}</div></div>
        <div class="stat"><p class="caption">Open findings</p><div class="display-md warn">{{ open_findings }}</div></div>
        <div class="stat"><p class="caption">Scans complete</p><div class="display-md ok">{{ done_count }}</div></div>
        <div class="stat"><p class="caption">Last scan</p><div class="display-md">{{ last_scan }}</div></div>
      </section>

      <div class="list-head"><span>Repository</span><span>Last scan</span><span>Severity</span><span></span></div>
      {% if rows %}{{ rows | safe }}{% else %}
      <div class="empty">
        <p class="caption">No repositories yet</p>
        <p class="body-md" style="margin:var(--md) 0 var(--lg);">Run your first scan to populate the overview.</p>
        <a class="btn" href="/scan-console">New scan</a>
      </div>
      {% endif %}
    </main>
  </div>
</body>
</html>"""


AUTH_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ 'Sign up' if mode == 'signup' else 'Sign in' }} — Sentinel</title>
  <style>""" + BUGATTI_CSS + """
    .auth-wrap{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;
      padding:var(--xl);background:radial-gradient(120% 90% at 50% 0%,#141414 0%,#000 55%);}
    .auth-card{width:100%;max-width:420px;}
    .toggle{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--hairline-strong);
      border-radius:var(--r-pill);overflow:hidden;margin-bottom:var(--xxl);}
    .toggle a{padding:12px 0;text-align:center;font-family:var(--font-mono);font-size:12px;letter-spacing:2px;
      text-transform:uppercase;color:var(--muted);text-decoration:none;}
    .toggle a.active{background:var(--on-dark);color:var(--on-primary);}
    .auth-card h1{text-align:center;margin-bottom:var(--xl);}
    .oauth-row{display:flex;gap:var(--sm);margin-bottom:var(--lg);}
    .oauth-row a{flex:1;padding:14px 0;border-color:var(--hairline-strong);color:var(--body);font-size:12px;letter-spacing:2px;}
    .divider{display:flex;align-items:center;gap:var(--md);margin-bottom:var(--lg);}
    .divider .line{flex:1;border-top:1px solid var(--hairline);}
    .error{font-family:var(--font-mono);font-size:11px;letter-spacing:1px;color:var(--warning);text-align:center;}
  </style>
</head>
<body>
  <nav class="top-nav"><a class="wordmark center" href="/">Sentinel</a></nav>
  <main class="auth-wrap">
    <div class="auth-card">
      <div class="toggle">
        <a href="/login" class="{{ 'active' if mode != 'signup' else '' }}">Sign in</a>
        <a href="/signup" class="{{ 'active' if mode == 'signup' else '' }}">Sign up</a>
      </div>

      <h1 class="display-md">{{ 'Create account' if mode == 'signup' else 'Sign in' }}</h1>

      {% if google_enabled or github_enabled %}
      <div class="oauth-row">
        {% if google_enabled %}<a class="btn" href="/auth/google">Google</a>{% endif %}
        {% if github_enabled %}<a class="btn" href="/auth/github">GitHub</a>{% endif %}
      </div>
      <div class="divider"><span class="line"></span><span class="caption">Or</span><span class="line"></span></div>
      {% endif %}

      <form method="POST" class="stack-lg">
        <div class="field">
          <label>Email</label>
          <input class="input" type="email" name="email" placeholder="you@company.com" required autofocus>
        </div>
        <div class="field">
          <label>Password</label>
          <input class="input" type="password" name="password" placeholder="••••••••" required>
        </div>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        <button type="submit" class="btn btn-full">{{ 'Create account' if mode == 'signup' else 'Continue' }}</button>
        <p class="caption center-text">Protected by hardware-key MFA</p>
      </form>
    </div>
  </main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Account settings page (Gmail auto-send connect/disconnect)
# ---------------------------------------------------------------------------
ACCOUNT_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Account Settings</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #080b10; color: #c8cdd8; font-family: 'Inter', sans-serif; padding: 2rem; }
    h1 { font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem; }
    a.back { color: #3b82f6; text-decoration: none; font-size: 0.85rem; }
    .card { background: #0e131c; border: 1px solid #1c2130; border-radius: 8px; padding: 1.25rem; margin-top: 1.5rem; max-width: 480px; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
    .label { font-size: 0.9rem; }
    .sub { font-size: 0.78rem; color: #8890a0; margin-top: 0.3rem; font-family: 'IBM Plex Mono', monospace; }
    .btn { padding: 0.5rem 0.9rem; border-radius: 4px; font-size: 0.82rem; text-decoration: none; cursor: pointer; border: none; }
    .btn-connect { background: #3b82f6; color: white; }
    .btn-disconnect { background: transparent; color: #ef4444; border: 1px solid #3a1f24; }
    form { display: inline; }
    .flash { font-size: 0.8rem; color: #3fb950; margin-bottom: 1rem; font-family: 'IBM Plex Mono', monospace; }
    .slack-form { display: block; margin-top: 0.75rem; }
    .slack-form input {
      width: 100%; padding: 0.5rem 0.6rem; margin-bottom: 0.5rem; border-radius: 4px;
      background: #161b26; border: 1px solid #1c2130; color: #c8cdd8; font-size: 0.82rem;
      font-family: 'IBM Plex Mono', monospace; box-sizing: border-box;
    }
    .help-link { color: #3b82f6; text-decoration: none; }
  </style>
</head>
<body>
  <a class="back" href="/scan-console">&larr; New scan</a>
  <h1 style="margin-top:1rem;">Account Settings</h1>

  {% with messages = get_flashed_messages() %}
    {% if messages %}{% for m in messages %}<div class="flash">{{ m }}</div>{% endfor %}{% endif %}
  {% endwith %}

  <div class="card">
    <div class="row">
      <div>
        <div class="label">Gmail auto-send</div>
        {% if connection %}
          <div class="sub">Connected as {{ connection.gmail_address }} -- reports email automatically when generated.</div>
        {% else %}
          <div class="sub">Not connected -- reports won't be emailed.</div>
        {% endif %}
      </div>
      {% if connection %}
        <form method="POST" action="/disconnect/gmail" onsubmit="return confirm('Disconnect Gmail? Reports will stop being emailed.');">
          <button class="btn btn-disconnect" type="submit">Disconnect</button>
        </form>
      {% elif google_enabled %}
        <a class="btn btn-connect" href="/connect/gmail">Connect Gmail</a>
      {% else %}
        <span class="sub">Google OAuth is not configured on this server.</span>
      {% endif %}
    </div>
  </div>

  <div class="card">
    <div class="row">
      <div>
        <div class="label">Slack notifications</div>
        {% if slack_connection %}
          <div class="sub">Connected{% if slack_connection.channel_label %} ({{ slack_connection.channel_label }}){% endif %} -- posts a summary when a scan fully completes (findings, fixes, PR).</div>
        {% else %}
          <div class="sub">Not connected -- paste an Incoming Webhook URL below. <a class="help-link" href="https://api.slack.com/messaging/webhooks" target="_blank">How to create one &rarr;</a></div>
        {% endif %}
      </div>
      {% if slack_connection %}
        <form method="POST" action="/disconnect/slack" onsubmit="return confirm('Disconnect Slack? Notifications will stop.');">
          <button class="btn btn-disconnect" type="submit">Disconnect</button>
        </form>
      {% endif %}
    </div>
    {% if not slack_connection %}
      <form class="slack-form" method="POST" action="/connect/slack">
        <input type="url" name="webhook_url" placeholder="https://hooks.slack.com/services/..." required>
        <input type="text" name="channel_label" placeholder="Label (optional, e.g. #security-alerts)">
        <button class="btn btn-connect" type="submit" style="width:100%;">Connect Slack</button>
      </form>
    {% endif %}
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Scan history page
# ---------------------------------------------------------------------------
HISTORY_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Scan History</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #080b10; color: #c8cdd8; font-family: 'Inter', sans-serif; padding: 2rem; }
    h1 { font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem; }
    a.back { color: #3b82f6; text-decoration: none; font-size: 0.85rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #1c2130; }
    th { color: #8890a0; font-weight: 500; }
    a { color: #3b82f6; text-decoration: none; }
  </style>
</head>
<body>
  <a class="back" href="/scan-console">&larr; New scan</a>
  <a class="back" href="/orgs" style="margin-left:1rem;">Organizations</a>
  <h1>Scan History</h1>
  <table>
    <tr><th>Repo</th><th>When</th><th>Status</th><th>Trigger</th><th>Findings</th><th>Fixes</th><th>Withheld</th><th>Report</th><th>SBOM</th><th>PR</th></tr>
    {{ rows | safe }}
  </table>
</body>
</html>"""


METRICS_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Scan Metrics</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #080b10; color: #c8cdd8; font-family: 'Inter', sans-serif; padding: 2rem; }
    h1 { font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem; }
    a.back { color: #3b82f6; text-decoration: none; font-size: 0.85rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem; }
    .card { background: #0e131c; border: 1px solid #1c2130; border-radius: 8px; padding: 1rem; }
    .card .label { color: #8890a0; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.03em; font-family: 'IBM Plex Mono', monospace; }
    .card .value { font-size: 1.6rem; margin-top: .35rem; font-family: 'IBM Plex Mono', monospace; }
    .ok { color: #3fb950; }
    .err { color: #f85149; }
    a { color: #3b82f6; }
  </style>
</head>
<body>
  <a class="back" href="/scan-console">&larr; New scan</a>
  <a class="back" href="/admin/scans" style="margin-left:1rem;">Raw scan list</a>
  <a class="back" href="/admin/metrics?format=json" style="margin-left:1rem;">JSON</a>
  <h1>Scan Metrics</h1>

  <div class="grid">
    <div class="card"><div class="label">Total scans</div><div class="value">{{ m.totals.all_scans }}</div></div>
    <div class="card"><div class="label">Last 24h</div><div class="value">{{ m.totals.last_24h }}</div></div>
    <div class="card"><div class="label">Last 7d</div><div class="value">{{ m.totals.last_7d }}</div></div>

    <div class="card"><div class="label">Running</div><div class="value">{{ m.by_status.running }}</div></div>
    <div class="card"><div class="label">Waiting approval</div><div class="value">{{ m.by_status.waiting_approval }}</div></div>
    <div class="card"><div class="label">Done</div><div class="value ok">{{ m.by_status.done }}</div></div>
    <div class="card"><div class="label">Error</div><div class="value err">{{ m.by_status.error }}</div></div>

    <div class="card"><div class="label">Avg duration</div>
      <div class="value">{{ m.performance.avg_duration_seconds if m.performance.avg_duration_seconds is not none else '--' }}s</div>
    </div>
    <div class="card"><div class="label">Error rate</div><div class="value">{{ m.performance.error_rate_pct }}%</div></div>

    <div class="card"><div class="label">Confirmed fixes</div><div class="value ok">{{ m.fix_pipeline.confirmed_fixes }}</div></div>
    <div class="card"><div class="label">Withheld fixes</div><div class="value err">{{ m.fix_pipeline.withheld_fixes }}</div></div>
    <div class="card"><div class="label">Confirm rate</div>
      <div class="value">{{ m.fix_pipeline.confirm_rate_pct if m.fix_pipeline.confirm_rate_pct is not none else '--' }}%</div>
    </div>

    <div class="card"><div class="label">Code findings (all-time)</div><div class="value">{{ m.findings.code_findings_total }}</div></div>
    <div class="card"><div class="label">Dep findings (all-time)</div><div class="value">{{ m.findings.dep_findings_total }}</div></div>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Repo management / dashboard page
# ---------------------------------------------------------------------------
REPOS_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>My Repos</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #080b10; color: #c8cdd8; font-family: 'Inter', sans-serif; padding: 2rem; }
    .top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
    h1 { font-size: 1.2rem; font-weight: 500; }
    a.new-scan {
      background: #3b82f6; color: white; text-decoration: none; padding: 0.5rem 1rem;
      border-radius: 4px; font-size: 0.85rem; font-weight: 500;
    }
    a.history-link { color: #8890a0; text-decoration: none; font-size: 0.85rem; margin-right: 1rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    th, td { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid #1c2130; vertical-align: middle; }
    th { color: #8890a0; font-weight: 500; }
    td.target { color: #6b7280; max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .btn-small {
      background: #161b25; border: 1px solid #2a3347; color: #c8cdd8; padding: 0.35rem 0.7rem;
      border-radius: 4px; font-size: 0.75rem; cursor: pointer; font-family: 'IBM Plex Mono', monospace;
    }
    .btn-small:hover { border-color: #3b82f6; }
    .btn-danger:hover { border-color: #ef4444; color: #ef4444; }
    .empty { color: #4a5268; font-size: 0.85rem; margin-top: 2rem; }
  </style>
</head>
<body>
  <div class="top">
    <h1>My Repos</h1>
    <div>
      <a class="history-link" href="/orgs">organizations</a>
      <a class="history-link" href="/history">history</a>
      <a class="history-link" href="/logout">log out</a>
      <a class="new-scan" href="/scan-console">+ New scan</a>
    </div>
  </div>

  {% if rows %}
  <table>
    <tr><th>Repo</th><th>Target</th><th>Scope</th><th>Last scan</th><th>Status</th><th>Findings</th><th>Schedule</th><th>Actions</th></tr>
    {{ rows | safe }}
  </table>
  {% else %}
  <div class="empty">No repos yet -- run a scan to add one, or <a href="/scan-console" style="color:#3b82f6;">start here</a>.</div>
  {% endif %}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Organizations list / create page
# ---------------------------------------------------------------------------
ORGS_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Organizations</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #080b10; color: #c8cdd8; font-family: 'Inter', sans-serif; padding: 2rem; }
    .top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
    h1 { font-size: 1.2rem; font-weight: 500; }
    h2 { font-size: 0.95rem; font-weight: 500; margin-bottom: 0.75rem; color: #e8ecf4; }
    a.link { color: #8890a0; text-decoration: none; font-size: 0.85rem; margin-right: 1rem; }
    a { color: #3b82f6; text-decoration: none; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    th, td { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid #1c2130; vertical-align: middle; }
    th { color: #8890a0; font-weight: 500; }
    .empty { color: #4a5268; font-size: 0.85rem; margin-top: 1rem; }
    .create-box {
      margin-top: 2.5rem; padding: 1.25rem; background: #0e1219; border: 1px solid #1c2130;
      border-radius: 8px; max-width: 420px;
    }
    .create-row { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
    input {
      flex: 1; padding: 0.6rem; background: #080b10; border: 1px solid #1c2130; border-radius: 4px;
      color: #c8cdd8; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    }
    button {
      padding: 0.6rem 1rem; background: #3b82f6; border: none; border-radius: 4px;
      color: white; font-weight: 500; cursor: pointer; font-size: 0.85rem; font-family: 'Inter', sans-serif;
    }
    button:hover { background: #2563eb; }
    .msg { font-size: 0.8rem; margin-top: 0.6rem; }
    .msg.error { color: #ef4444; }
    .msg.ok { color: #22c55e; }
  </style>
</head>
<body>
  <div class="top">
    <h1>Organizations</h1>
    <div>
      <a class="link" href="/repos">my repos</a>
      <a class="link" href="/history">history</a>
      <a class="link" href="/logout">log out</a>
    </div>
  </div>

  {% if rows %}
  <table>
    <tr><th>Name</th><th>Your role</th><th>Members</th><th></th></tr>
    {{ rows | safe }}
  </table>
  {% else %}
  <div class="empty">You aren't a member of any organization yet -- create one below to start sharing repos with a team.</div>
  {% endif %}

  <div class="create-box">
    <h2>Create an organization</h2>
    <div class="create-row">
      <input type="text" id="org-name" placeholder="Acme Corp">
      <button onclick="createOrg()">Create</button>
    </div>
    <div class="msg" id="create-msg"></div>
  </div>

<script>
  async function createOrg() {
    const name = document.getElementById('org-name').value.trim();
    const msg = document.getElementById('create-msg');
    msg.textContent = '';
    msg.className = 'msg';
    if (!name) { msg.textContent = 'Enter a name first.'; msg.className = 'msg error'; return; }

    try {
      const res = await fetch('/organizations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      const data = await res.json();
      if (!res.ok) { msg.textContent = data.error || 'Failed to create organization.'; msg.className = 'msg error'; return; }
      window.location.href = '/orgs/' + data.id;
    } catch (e) {
      msg.textContent = 'Failed to create organization: ' + e.message;
      msg.className = 'msg error';
    }
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Single-organization management page: members, roles, invite
# ---------------------------------------------------------------------------
ORG_DETAIL_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ org.name }} — Organization</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #080b10; color: #c8cdd8; font-family: 'Inter', sans-serif; padding: 2rem; }
    .top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem; }
    h1 { font-size: 1.2rem; font-weight: 500; }
    h2 { font-size: 0.95rem; font-weight: 500; margin-bottom: 0.75rem; color: #e8ecf4; }
    a.link { color: #8890a0; text-decoration: none; font-size: 0.85rem; margin-right: 1rem; }
    a.back { color: #3b82f6; text-decoration: none; font-size: 0.85rem; }
    .role-tag {
      display: inline-block; margin-top: 0.5rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
      color: #8890a0; border: 1px solid #1c2130; padding: 2px 8px; border-radius: 3px;
    }
    table { width: 100%; border-collapse: collapse; margin-top: 1.25rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    th, td { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid #1c2130; vertical-align: middle; }
    th { color: #8890a0; font-weight: 500; }
    select {
      background: #080b10; border: 1px solid #1c2130; border-radius: 4px; color: #c8cdd8;
      font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; padding: 0.35rem 0.5rem;
    }
    .btn-small {
      background: #161b25; border: 1px solid #2a3347; color: #c8cdd8; padding: 0.35rem 0.7rem;
      border-radius: 4px; font-size: 0.75rem; cursor: pointer; font-family: 'IBM Plex Mono', monospace;
    }
    .btn-small:hover { border-color: #3b82f6; }
    .btn-danger:hover { border-color: #ef4444; color: #ef4444; }
    .invite-box {
      margin-top: 2.5rem; padding: 1.25rem; background: #0e1219; border: 1px solid #1c2130;
      border-radius: 8px; max-width: 480px;
    }
    .invite-row { display: flex; gap: 0.5rem; }
    input[type=email] {
      flex: 1; padding: 0.6rem; background: #080b10; border: 1px solid #1c2130; border-radius: 4px;
      color: #c8cdd8; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    }
    .invite-row button {
      padding: 0.6rem 1rem; background: #3b82f6; border: none; border-radius: 4px;
      color: white; font-weight: 500; cursor: pointer; font-size: 0.85rem; font-family: 'Inter', sans-serif;
    }
    .invite-row button:hover { background: #2563eb; }
    .invite-msg { font-size: 0.8rem; margin-top: 0.6rem; }
    .invite-msg.error { color: #ef4444; }
    .invite-msg.ok { color: #22c55e; }
  </style>
</head>
<body>
  <div class="top">
    <a class="back" href="/orgs">&larr; Organizations</a>
    <div>
      <a class="link" href="/repos">my repos</a>
      <a class="link" href="/history">history</a>
      <a class="link" href="/logout">log out</a>
    </div>
  </div>
  <h1 style="margin-top:1rem;">{{ org.name }}</h1>
  <div class="role-tag">your role: {{ my_role }}</div>

  <table>
    <tr><th>Email</th><th>Role</th><th>Joined</th><th></th></tr>
    {{ member_rows | safe }}
  </table>

  {{ invite_form | safe }}

  {% if invite_rows %}
  <div class="invite-box" style="max-width:640px;">
    <h2>Pending invites</h2>
    <table>
      <tr><th>Email</th><th>Role</th><th>Sent</th><th></th></tr>
      {{ invite_rows | safe }}
    </table>
  </div>
  {% endif %}

  {{ settings_box | safe }}

<script>
  async function inviteMember(orgId) {
    const email = document.getElementById('invite-email').value.trim();
    const role = document.getElementById('invite-role').value;
    const msg = document.getElementById('invite-msg');
    msg.textContent = '';
    msg.className = 'invite-msg';
    if (!email) { msg.textContent = 'Enter an email first.'; msg.className = 'invite-msg error'; return; }

    try {
      const res = await fetch(`/organizations/${orgId}/members`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, role })
      });
      const data = await res.json();
      if (!res.ok) { msg.textContent = data.error || 'Failed to add member.'; msg.className = 'invite-msg error'; return; }
      window.location.reload();
    } catch (e) {
      msg.textContent = 'Failed to add member: ' + e.message;
      msg.className = 'invite-msg error';
    }
  }

  async function changeRole(orgId, userId, newRole) {
    try {
      const res = await fetch(`/organizations/${orgId}/members/${userId}/role`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole })
      });
      const data = await res.json();
      if (!res.ok) { alert(data.error || 'Failed to change role.'); window.location.reload(); return; }
      window.location.reload();
    } catch (e) {
      alert('Failed to change role: ' + e.message);
    }
  }

  async function removeMember(orgId, userId, email) {
    if (!confirm(`Remove ${email} from this organization?`)) return;
    try {
      const res = await fetch(`/organizations/${orgId}/members/${userId}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) { alert(data.error || 'Failed to remove member.'); return; }
      window.location.reload();
    } catch (e) {
      alert('Failed to remove member: ' + e.message);
    }
  }

  async function revokeInvite(orgId, inviteId, email) {
    if (!confirm(`Revoke the pending invite for ${email}?`)) return;
    try {
      const res = await fetch(`/organizations/${orgId}/invites/${inviteId}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) { alert(data.error || 'Failed to revoke invite.'); return; }
      window.location.reload();
    } catch (e) {
      alert('Failed to revoke invite: ' + e.message);
    }
  }

  async function saveSettings(orgId) {
    const freq = document.getElementById('default-freq').value;
    const msg = document.getElementById('settings-msg');
    msg.textContent = '';
    msg.className = 'invite-msg';
    try {
      const res = await fetch(`/organizations/${orgId}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ default_scan_frequency: freq })
      });
      const data = await res.json();
      if (!res.ok) { msg.textContent = data.error || 'Failed to save settings.'; msg.className = 'invite-msg error'; return; }
      msg.textContent = 'Saved.';
      msg.className = 'invite-msg ok';
    } catch (e) {
      msg.textContent = 'Failed to save settings: ' + e.message;
      msg.className = 'invite-msg error';
    }
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Per-repo access management page: restrict a repo + grant/revoke access
# and push rights for individual org members.
# ---------------------------------------------------------------------------
REPO_ACCESS_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ repo.name }} — Access</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #080b10; color: #c8cdd8; font-family: 'Inter', sans-serif; padding: 2rem; }
    h1 { font-size: 1.2rem; font-weight: 500; }
    h2 { font-size: 0.95rem; font-weight: 500; margin-bottom: 0.75rem; color: #e8ecf4; }
    a.back { color: #3b82f6; text-decoration: none; font-size: 0.85rem; }
    .sub { color: #8890a0; font-size: 0.8rem; margin-top: 0.3rem; }
    .toggle-box {
      margin-top: 1.5rem; padding: 1.25rem; background: #0e1219; border: 1px solid #1c2130;
      border-radius: 8px; max-width: 560px;
    }
    label { font-size: 0.85rem; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    th, td { text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid #1c2130; vertical-align: middle; }
    th { color: #8890a0; font-weight: 500; }
    select, input {
      background: #080b10; border: 1px solid #1c2130; border-radius: 4px; color: #c8cdd8;
      font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; padding: 0.4rem 0.5rem;
    }
    .btn-small {
      background: #161b25; border: 1px solid #2a3347; color: #c8cdd8; padding: 0.35rem 0.7rem;
      border-radius: 4px; font-size: 0.75rem; cursor: pointer; font-family: 'IBM Plex Mono', monospace;
    }
    .btn-small:hover { border-color: #3b82f6; }
    .btn-danger:hover { border-color: #ef4444; color: #ef4444; }
    .grant-row { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.75rem; }
    .grant-msg { font-size: 0.8rem; margin-top: 0.6rem; }
    .grant-msg.error { color: #ef4444; }
    .grant-msg.ok { color: #22c55e; }
  </style>
</head>
<body>
  <a class="back" href="/repos">&larr; My repos</a>
  <h1 style="margin-top:1rem;">{{ repo.name }}</h1>
  <div class="sub">{{ org.name }} &middot; {{ repo.target }}</div>

  <div class="toggle-box">
    <h2>Restrict visibility</h2>
    <label>
      <input type="checkbox" id="restricted-toggle" {{ "checked" if repo.restricted else "" }}
             onchange="toggleRestricted({{ repo.id }}, this.checked)">
      Only members explicitly granted access below can see this repo (owners/admins always can)
    </label>
  </div>

  <div class="toggle-box">
    <h2>Access grants</h2>
    <table>
      <tr><th>Email</th><th>Can push / open PR</th><th></th></tr>
      {{ grant_rows | safe }}
    </table>

    <div class="grant-row">
      <select id="grant-email">{{ member_options | safe }}</select>
      <label style="gap:0.35rem;"><input type="checkbox" id="grant-can-push"> can push</label>
      <button class="btn-small" onclick="addAccess({{ repo.id }})">Grant</button>
    </div>
    <div class="grant-msg" id="grant-msg"></div>
  </div>

<script>
  async function toggleRestricted(repoId, checked) {
    try {
      const res = await fetch(`/repos/${repoId}/access`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ restricted: checked })
      });
      if (!res.ok) { const d = await res.json(); alert(d.error || 'Failed to update.'); window.location.reload(); }
    } catch (e) {
      alert('Failed to update: ' + e.message);
    }
  }

  async function addAccess(repoId) {
    const emailSelect = document.getElementById('grant-email');
    const email = emailSelect.value;
    const canPush = document.getElementById('grant-can-push').checked;
    const msg = document.getElementById('grant-msg');
    msg.textContent = '';
    msg.className = 'grant-msg';
    if (!email) { msg.textContent = 'No eligible members left to grant.'; msg.className = 'grant-msg error'; return; }

    try {
      const res = await fetch(`/repos/${repoId}/access`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, can_push: canPush })
      });
      const data = await res.json();
      if (!res.ok) { msg.textContent = data.error || 'Failed to grant access.'; msg.className = 'grant-msg error'; return; }
      window.location.reload();
    } catch (e) {
      msg.textContent = 'Failed to grant access: ' + e.message;
      msg.className = 'grant-msg error';
    }
  }

  async function revokeAccess(repoId, userId, email) {
    if (!confirm(`Revoke access for ${email}?`)) return;
    try {
      const res = await fetch(`/repos/${repoId}/access/${userId}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) { alert(data.error || 'Failed to revoke access.'); return; }
      window.location.reload();
    } catch (e) {
      alert('Failed to revoke access: ' + e.message);
    }
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# The entire frontend UI as a single HTML string
# ---------------------------------------------------------------------------
HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vulnerability Scanner</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #080b10;
      --surface:   #0e1219;
      --border:    #1c2130;
      --border-hi: #2a3347;
      --text:      #c8cdd8;
      --text-dim:  #4a5268;
      --text-hi:   #e8ecf4;
      --accent:    #3b82f6;
      --accent-dim:#1d4ed8;
      --green:     #22c55e;
      --red:       #ef4444;
      --yellow:    #f59e0b;
      --mono:      'IBM Plex Mono', monospace;
      --sans:      'Inter', sans-serif;
    }

    html, body {
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.6;
    }

    /* ── Layout ── */
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    header {
      border-bottom: 1px solid var(--border);
      padding: 0 2rem;
      height: 52px;
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .logo {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--text-hi);
      letter-spacing: .04em;
      display: flex;
      align-items: center;
      gap: .5rem;
    }

    .logo-dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent);
    }

    .header-tag {
      font-family: var(--mono);
      font-size: 10px;
      color: var(--text-dim);
      border: 1px solid var(--border);
      padding: 2px 8px;
      border-radius: 3px;
    }

    main {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 3rem 1rem;
    }

    .card {
      width: 100%;
      max-width: 560px;
    }

    /* ── Hero text ── */
    .hero {
      margin-bottom: 2.5rem;
    }

    .hero h1 {
      font-size: 1.75rem;
      font-weight: 600;
      color: var(--text-hi);
      letter-spacing: -.02em;
      line-height: 1.2;
      margin-bottom: .5rem;
    }

    .hero p {
      color: var(--text-dim);
      font-size: 13px;
    }

    /* ── Tab switcher ── */
    .tabs {
      display: flex;
      gap: .25rem;
      margin-bottom: 1rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: .75rem;
    }

    .tab {
      font-family: var(--mono);
      font-size: 11px;
      padding: 5px 14px;
      border-radius: 4px;
      border: 1px solid transparent;
      cursor: pointer;
      color: var(--text-dim);
      background: none;
      transition: all .15s;
      letter-spacing: .03em;
    }

    .tab.active {
      background: var(--surface);
      border-color: var(--border-hi);
      color: var(--text-hi);
    }

    /* ── Panels ── */
    .panel { display: none; }
    .panel.active { display: block; }

    /* ── URL input ── */
    .input-row {
      display: flex;
      gap: .5rem;
    }

    .url-input {
      flex: 1;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 14px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--text-hi);
      outline: none;
      transition: border-color .15s;
    }

    .url-input::placeholder { color: var(--text-dim); }
    .url-input:focus { border-color: var(--accent); }

    /* ── Drop zone ── */
    .dropzone {
      border: 1px dashed var(--border-hi);
      border-radius: 8px;
      padding: 3rem 2rem;
      text-align: center;
      cursor: pointer;
      transition: all .2s;
      position: relative;
    }

    .dropzone:hover, .dropzone.drag-over {
      border-color: var(--accent);
      background: rgba(59,130,246,.04);
    }

    .dropzone input[type=file] {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }

    .drop-icon {
      font-size: 2rem;
      margin-bottom: .75rem;
      opacity: .4;
    }

    .drop-label {
      color: var(--text-dim);
      font-size: 13px;
      line-height: 1.8;
    }

    .drop-label strong {
      color: var(--accent);
      font-weight: 500;
    }

    .drop-hint {
      margin-top: .5rem;
      font-size: 11px;
      color: var(--text-dim);
      font-family: var(--mono);
    }

    /* ── Scan button ── */
    .btn-scan {
      width: 100%;
      margin-top: 1rem;
      padding: 11px;
      background: var(--accent);
      border: none;
      border-radius: 6px;
      color: #fff;
      font-family: var(--sans);
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: background .15s;
      letter-spacing: .01em;
    }

    .btn-scan:hover { background: var(--accent-dim); }
    .btn-scan:disabled { opacity: .4; cursor: not-allowed; }

    /* ── Progress panel ── */
    .progress-panel {
      display: none;
      margin-top: 2rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }

    .progress-panel.visible { display: block; }

    .progress-header {
      padding: .75rem 1rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: .5rem;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text-dim);
    }

    .pulse {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--accent);
      animation: pulse 1.4s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: .3; }
    }

    .progress-steps {
      padding: .75rem 1rem;
      display: flex;
      flex-direction: column;
      gap: .35rem;
    }

    .step {
      display: flex;
      align-items: center;
      gap: .75rem;
      padding: .4rem .5rem;
      border-radius: 4px;
      transition: background .2s;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text-dim);
    }

    .step.done {
      color: var(--text);
    }

    .step.active {
      background: rgba(59,130,246,.06);
      color: var(--text-hi);
    }

    .step-icon {
      width: 16px;
      text-align: center;
      flex-shrink: 0;
    }

    .step.done .step-icon::before   { content: "✓"; color: var(--green); }
    .step.active .step-icon::before { content: "›"; color: var(--accent); }
    .step.wait .step-icon::before   { content: "·"; color: var(--text-dim); }
    .step.error .step-icon::before  { content: "✗"; color: var(--red); }

    /* ── Approval panel ── */
    .approval-panel {
      display: none;
      margin-top: 1rem;
      background: var(--surface);
      border: 1px solid var(--yellow);
      border-radius: 8px;
      overflow: hidden;
    }

    .approval-panel.visible { display: block; }

    .approval-header {
      padding: .75rem 1rem;
      border-bottom: 1px solid var(--border);
      font-family: var(--mono);
      font-size: 11px;
      color: var(--yellow);
      letter-spacing: .03em;
    }

    .approval-body {
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: .5rem;
    }

    .approval-row {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text);
      padding: .3rem 0;
      border-bottom: 1px solid var(--border);
    }

    .approval-row .score-pass { color: var(--green); }
    .approval-row .score-warn { color: var(--yellow); }
    .approval-row .score-fail { color: var(--red); }

    .approval-question {
      font-size: 12px;
      color: var(--text-hi);
      margin: .5rem 0;
    }

    .approval-btns {
      display: flex;
      gap: .5rem;
      padding: 0 1rem 1rem;
    }

    .btn-approve, .btn-reject {
      flex: 1;
      padding: 9px;
      border-radius: 6px;
      font-family: var(--sans);
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      border: none;
      transition: opacity .15s;
    }

    .btn-approve { background: var(--green); color: #06210f; }
    .btn-reject  { background: var(--border-hi); color: var(--text); }
    .btn-approve:hover, .btn-reject:hover { opacity: .85; }
    .btn-approve:disabled, .btn-reject:disabled { opacity: .4; cursor: not-allowed; }

    /* ── Result panel ── */
    .result-panel {
      display: none;
      margin-top: 1rem;
      padding: 1.25rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
    }

    .result-panel.visible { display: block; }

    .result-stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: .75rem;
      margin-bottom: 1.25rem;
    }

    .stat {
      text-align: center;
      padding: .75rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
    }

    .stat-number {
      font-family: var(--mono);
      font-size: 1.5rem;
      font-weight: 500;
      color: var(--text-hi);
      line-height: 1;
      margin-bottom: .25rem;
    }

    .stat-label {
      font-size: 11px;
      color: var(--text-dim);
    }

    .btn-report {
      display: block;
      width: 100%;
      padding: 10px;
      background: var(--bg);
      border: 1px solid var(--accent);
      border-radius: 6px;
      color: var(--accent);
      font-family: var(--mono);
      font-size: 11px;
      text-align: center;
      cursor: pointer;
      text-decoration: none;
      transition: all .15s;
      letter-spacing: .04em;
      margin-bottom: .5rem;
    }

    .btn-report:hover {
      background: rgba(59,130,246,.08);
    }

    .btn-pr {
      display: block;
      width: 100%;
      padding: 10px;
      background: var(--bg);
      border: 1px solid var(--green);
      border-radius: 6px;
      color: var(--green);
      font-family: var(--mono);
      font-size: 11px;
      text-align: center;
      cursor: pointer;
      text-decoration: none;
      transition: all .15s;
      letter-spacing: .04em;
      margin-bottom: .5rem;
    }

    .btn-pr:hover {
      background: rgba(34,197,94,.08);
    }

    .btn-new {
      display: block;
      width: 100%;
      margin-top: .5rem;
      padding: 9px;
      background: none;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text-dim);
      font-family: var(--mono);
      font-size: 11px;
      text-align: center;
      cursor: pointer;
      transition: all .15s;
      letter-spacing: .04em;
    }

    .btn-new:hover {
      border-color: var(--border-hi);
      color: var(--text);
    }

    /* ── Diff viewer modal ── */
    .btn-fixes {
      display: block;
      width: 100%;
      margin-top: .5rem;
      padding: 9px;
      background: none;
      border: 1px solid var(--accent);
      border-radius: 6px;
      color: var(--accent);
      font-family: var(--mono);
      font-size: 11px;
      text-align: center;
      cursor: pointer;
      transition: all .15s;
      letter-spacing: .04em;
    }
    .btn-fixes:hover { background: rgba(59,130,246,.1); }

    .diff-modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,.6);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 2rem 1rem;
    }
    .diff-modal-backdrop.visible { display: flex; }

    .diff-modal {
      background: var(--surface);
      border: 1px solid var(--border-hi);
      border-radius: 10px;
      width: 100%;
      max-width: 900px;
      max-height: 85vh;
      display: flex;
      overflow: hidden;
    }

    .diff-modal-sidebar {
      width: 220px;
      flex-shrink: 0;
      border-right: 1px solid var(--border);
      overflow-y: auto;
      padding: .5rem;
    }

    .diff-file-item {
      padding: .5rem .6rem;
      border-radius: 5px;
      font-family: var(--mono);
      font-size: 10.5px;
      color: var(--text-dim);
      cursor: pointer;
      word-break: break-all;
      margin-bottom: 2px;
    }
    .diff-file-item:hover { background: var(--border); color: var(--text); }
    .diff-file-item.active { background: var(--accent-dim); color: var(--text-hi); }

    .diff-modal-body {
      flex: 1;
      overflow-y: auto;
      padding: 1.25rem;
    }

    .diff-modal-close {
      position: absolute;
      top: .75rem;
      right: 1rem;
      background: none;
      border: none;
      color: var(--text-dim);
      font-size: 20px;
      cursor: pointer;
      line-height: 1;
    }
    .diff-modal-close:hover { color: var(--text); }

    .diff-file-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: .5rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }
    .diff-file-path {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--text-hi);
      word-break: break-all;
    }

    .diff-section-label {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: .06em;
      color: var(--text-dim);
      text-transform: uppercase;
      margin: 1rem 0 .4rem;
    }

    .diff-checklist { display: flex; flex-direction: column; gap: 3px; }
    .diff-check {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: .5rem;
    }
    .diff-check-badge {
      font-size: 9px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 4px;
      letter-spacing: .03em;
      flex-shrink: 0;
    }
    .check-pass { background: rgba(34,197,94,.15); color: var(--green); }
    .check-fail { background: rgba(239,68,68,.15); color: var(--red); }
    .check-warn { background: rgba(245,158,11,.15); color: var(--yellow); }
    .check-skip { background: var(--border); color: var(--text-dim); }

    .diff-problem {
      font-family: var(--mono);
      font-size: 10.5px;
      color: var(--text-dim);
      padding-left: 1rem;
      margin-top: 2px;
    }
    .diff-problem-sev { color: var(--yellow); }

    .diff-view {
      font-family: var(--mono);
      font-size: 11px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: .75rem;
      overflow-x: auto;
      white-space: pre;
    }
    .diff-line { padding: 1px 4px; }
    .diff-add { background: rgba(34,197,94,.12); color: var(--green); }
    .diff-del { background: rgba(239,68,68,.12); color: var(--red); }
    .diff-ctx { color: var(--text-dim); }
    .diff-hunk { color: var(--accent); margin: 4px 0; }

    /* ── Error state ── */
    .error-box {
      display: none;
      margin-top: 1rem;
      padding: 1rem;
      background: rgba(239,68,68,.06);
      border: 1px solid rgba(239,68,68,.2);
      border-radius: 8px;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--red);
    }

    .error-box.visible { display: block; }

    footer {
      border-top: 1px solid var(--border);
      padding: .75rem 2rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      font-family: var(--mono);
      font-size: 10px;
      color: var(--text-dim);
    }
  </style>
</head>
<body>
<div class="shell">

  <header>
    <div class="logo">
      <div class="logo-dot"></div>
      vuln-agent
    </div>
    <span class="header-tag">bandit · pip-audit · llama-3 · critic</span>
    <span style="margin-left:auto; font-family:var(--mono); font-size:11px; display:flex; gap:1rem; align-items:center;">
      <a href="/repos" style="color:var(--text-dim);">my repos</a>
      <a href="/orgs" style="color:var(--text-dim);">organizations</a>
      <a href="/history" style="color:var(--text-dim);">history</a>
      <a href="/account" style="color:var(--text-dim);">account</a>
      <a href="/logout" style="color:var(--text-dim);">log out</a>
    </span>
  </header>

  <main>
    <div class="card">

      <div class="hero">
        <h1>Scan a Python repo<br>for vulnerabilities</h1>
        <p>Paste a GitHub URL or drop a zipped project folder.<br>Get a report, AI-generated fixes, and an optional PR — with your approval at each step.</p>
      </div>

      <!-- Tab switcher -->
      <div class="tabs">
        <button class="tab active" onclick="switchTab('url')">GitHub URL</button>
        <button class="tab" onclick="switchTab('upload')">Upload folder</button>
      </div>

      <!-- Personal vs. team scope -->
      <div class="input-row" style="margin-bottom: 1rem;">
        <select id="org-select" class="url-input" style="cursor:pointer;">
          <option value="">Personal (only visible to you)</option>
          {{ org_options | safe }}
        </select>
      </div>

      <!-- URL panel -->
      <div class="panel active" id="panel-url">
        <div class="input-row">
          <input class="url-input" id="url-input" type="text"
                 placeholder="https://github.com/user/repo"
                 onkeydown="if(event.key==='Enter') startScan()">
        </div>
      </div>

      <!-- Upload panel -->
      <div class="panel" id="panel-upload">
        <div class="dropzone" id="dropzone"
             ondragover="onDragOver(event)"
             ondragleave="onDragLeave(event)"
             ondrop="onDrop(event)">
          <input type="file" id="file-input" accept=".zip" onchange="onFileSelect(event)">
          <div class="drop-icon">⬡</div>
          <div class="drop-label">
            <strong>Drop your project folder here</strong><br>
            or click to browse
          </div>
          <div class="drop-hint" id="file-name">zip your folder first · max 100MB</div>
        </div>
      </div>

      <button class="btn-scan" id="btn-scan" onclick="startScan()">
        Run security scan
      </button>

      <!-- Progress -->
      <div class="progress-panel" id="progress-panel">
        <div class="progress-header">
          <div class="pulse" id="pulse-dot"></div>
          <span id="progress-label">Initialising scan...</span>
        </div>
        <div class="progress-steps" id="steps-container"></div>
      </div>

      <!-- Approval (shown twice: report checkpoint, then PR checkpoint) -->
      <div class="approval-panel" id="approval-panel">
        <div class="approval-header" id="approval-header">HUMAN APPROVAL REQUIRED</div>
        <div class="approval-body" id="approval-body"></div>
        <div class="approval-btns">
          <button class="btn-approve" id="btn-approve" onclick="sendApproval('approve')">Approve</button>
          <button class="btn-reject" id="btn-reject" onclick="sendApproval('reject')">Reject</button>
        </div>
        <div class="approval-btns" style="padding-top:0;">
          <button class="btn-reject" id="btn-approve-all" onclick="sendApproval('approve_all')"
                  style="display:none; color:var(--yellow); border:1px solid var(--yellow); background:none;">
            Approve All (include withheld, unconfirmed fixes)
          </button>
        </div>
      </div>

      <!-- Result -->
      <div class="result-panel" id="result-panel">
        <div class="result-stats">
          <div class="stat">
            <div class="stat-number" id="stat-code">—</div>
            <div class="stat-label">code issues</div>
          </div>
          <div class="stat">
            <div class="stat-number" id="stat-dep">—</div>
            <div class="stat-label">dep vulns</div>
          </div>
          <div class="stat">
            <div class="stat-number" id="stat-total">—</div>
            <div class="stat-label">total findings</div>
          </div>
        </div>
        <a class="btn-report" id="btn-report" href="#" target="_blank">
          VIEW FULL REPORT →
        </a>
        <a class="btn-pr" id="btn-pr" href="#" target="_blank" style="display:none;">
          VIEW PULL REQUEST →
        </a>
        <button class="btn-fixes" id="btn-fixes" style="display:none;" onclick="openDiffModal()">
          VIEW FIXES &amp; DIFFS
        </button>
        <button class="btn-new" onclick="resetUI()">← Scan another repo</button>
      </div>

      <!-- Diff viewer modal -->
      <div class="diff-modal-backdrop" id="diff-modal-backdrop" onclick="if(event.target===this) closeDiffModal()">
        <div class="diff-modal">
          <button class="diff-modal-close" onclick="closeDiffModal()">&times;</button>
          <div class="diff-modal-sidebar" id="diff-modal-sidebar"></div>
          <div class="diff-modal-body" id="diff-modal-body">
            <div class="diff-line diff-ctx">Select a file on the left to view its diff.</div>
          </div>
        </div>
      </div>

      <!-- Error -->
      <div class="error-box" id="error-box"></div>

    </div>
  </main>

  <footer>
    <span>Vulnerability Finder Agent</span>
    <span>·</span>
    <span>Python repos only</span>
    <span>·</span>
    <span>Public GitHub repos only</span>
  </footer>

</div>

<script>
  let activeTab = 'url';
  let selectedFile = null;
  let currentScanId = null;

  const STEP_DEFS = [
    { node: 'router',       label: 'Detecting input type'     },
    { node: 'github_fetch', label: 'Cloning repository'       },
    { node: 'local_read',   label: 'Reading local folder'     },
    { node: 'code_scan',    label: 'Running static analysis'  },
    { node: 'dep_scan',     label: 'Scanning dependencies'    },
    { node: 'cve_enrich',   label: 'Enriching with CVE data'  },
    { node: 'human_review', label: 'Recording report decision'},
    { node: 'report',       label: 'Generating AI report'     },
    { node: 'fix_generate', label: 'Generating fixes'         },
    { node: 'pr_review',    label: 'Recording PR decision'    },
    { node: 'pr_generate',  label: 'Opening pull request'     },
  ];

  function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.tab').forEach((t, i) => {
      t.classList.toggle('active', (i === 0 && tab === 'url') || (i === 1 && tab === 'upload'));
    });
    document.getElementById('panel-url').classList.toggle('active', tab === 'url');
    document.getElementById('panel-upload').classList.toggle('active', tab === 'upload');
  }

  function onDragOver(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.add('drag-over');
  }

  function onDragLeave(e) {
    document.getElementById('dropzone').classList.remove('drag-over');
  }

  function onDrop(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  }

  function onFileSelect(e) {
    const file = e.target.files[0];
    if (file) setFile(file);
  }

  function setFile(file) {
    selectedFile = file;
    document.getElementById('file-name').textContent = file.name;
  }

  function buildSteps(completedNodes) {
    const container = document.getElementById('steps-container');
    container.innerHTML = '';

    const completedSet = new Set(completedNodes.map(n => n.node));

    const showGithub = completedSet.has('github_fetch');
    const showLocal  = completedSet.has('local_read');

    STEP_DEFS.forEach(def => {
      if (def.node === 'github_fetch' && !showGithub) return;
      if (def.node === 'local_read'   && !showLocal)  return;

      const completed = completedSet.has(def.node);
      const isActive  = !completed && completedNodes.length > 0 &&
                        STEP_DEFS.indexOf(def) === completedNodes.length;

      const div = document.createElement('div');
      div.className = `step ${completed ? 'done' : isActive ? 'active' : 'wait'}`;
      div.id = `step-${def.node}`;
      div.innerHTML = `<span class="step-icon"></span><span>${def.label}</span>`;
      container.appendChild(div);
    });
  }

  async function startScan() {
    const btn = document.getElementById('btn-scan');
    btn.disabled = true;
    document.getElementById('error-box').classList.remove('visible');
    document.getElementById('result-panel').classList.remove('visible');
    document.getElementById('approval-panel').classList.remove('visible');

    let scanId;

    try {
      const orgId = document.getElementById('org-select').value;

      if (activeTab === 'url') {
        const url = document.getElementById('url-input').value.trim();
        if (!url) { showError('Please enter a GitHub URL.'); btn.disabled = false; return; }

        const payload = { url };
        if (orgId) payload.organization_id = orgId;

        const res = await fetch('/scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) { showError(data.error); btn.disabled = false; return; }
        scanId = data.scan_id;

      } else {
        if (!selectedFile) { showError('Please select a zip file.'); btn.disabled = false; return; }
        const form = new FormData();
        form.append('file', selectedFile);
        if (orgId) form.append('organization_id', orgId);

        const res = await fetch('/scan', { method: 'POST', body: form });
        const data = await res.json();
        if (data.error) { showError(data.error); btn.disabled = false; return; }
        scanId = data.scan_id;
      }
    } catch (e) {
      showError('Failed to start scan: ' + e.message);
      btn.disabled = false;
      return;
    }

    currentScanId = scanId;
    showProgress(scanId);
  }

  function showProgress(scanId) {
    const panel = document.getElementById('progress-panel');
    panel.classList.add('visible');
    document.getElementById('progress-label').textContent = 'Scan running...';

    buildSteps([]);

    const completedNodes = [];
    const es = new EventSource(`/stream/${scanId}`);

    es.onmessage = (e) => {
      const event = JSON.parse(e.data);

      if (event.type === 'node_complete') {
        const { node, label, step, total } = event.data;
        completedNodes.push({ node, label });
        buildSteps(completedNodes);
        document.getElementById('progress-label').textContent =
          `Step ${step} of ${total} — ${label}`;
      }

      if (event.type === 'approval_required') {
        showApproval(event.data);
      }

      if (event.type === 'done') {
        es.close();
        document.getElementById('pulse-dot').style.background = 'var(--green)';
        document.getElementById('pulse-dot').style.animation = 'none';
        document.getElementById('progress-label').textContent = 'Scan complete';
        showResult(event.data);
      }

      if (event.type === 'error') {
        es.close();
        document.getElementById('btn-scan').disabled = false;
        document.getElementById('approval-panel').classList.remove('visible');
        showError(event.data.message);
      }

      if (event.type === 'close') {
        es.close();
      }
    };

    es.onerror = () => {
      es.close();
    };
  }

  function scoreClass(score) {
    if (score === null || score === undefined) return '';
    if (score >= 8) return 'score-pass';
    if (score >= 5) return 'score-warn';
    return 'score-fail';
  }

  function showApproval(payload) {
    const panel = document.getElementById('approval-panel');
    const body = document.getElementById('approval-body');
    const header = document.getElementById('approval-header');
    body.innerHTML = '';

    if (payload.checkpoint === 'report_approval') {
      header.textContent = 'APPROVAL REQUIRED — REPORT GENERATION';
      body.innerHTML += `<div class="approval-row">repo: ${payload.repo_name}</div>`;
      body.innerHTML += `<div class="approval-row">total findings: ${payload.total_findings}</div>`;
      body.innerHTML += `<div class="approval-row">severity: ${JSON.stringify(payload.severity_counts)}</div>`;
      body.innerHTML += `<div class="approval-question">Proceed to AI report generation?</div>`;

    } else if (payload.checkpoint === 'pr_approval') {
      header.textContent = 'APPROVAL REQUIRED — PULL REQUEST';
      body.innerHTML += `<div class="approval-row">repo: ${payload.repo_name}</div>`;
      (payload.code_fixes || []).forEach(f => {
        const cls = scoreClass(f.critic_score);
        body.innerHTML += `<div class="approval-row">${f.path} (${f.findings_addressed} finding(s)) <span class="${cls}">[critic ${f.critic_score}/10 ${f.critic_verdict}]</span></div>`;
      });
      (payload.dependency_fixes || []).forEach(f => {
        body.innerHTML += `<div class="approval-row">[${f.type}] ${f.package}: ${f.old} → ${f.new}</div>`;
      });

      // Withheld fixes -- previously computed by the backend but never
      // rendered here, so a web user approving a PR never saw the same
      // "these fixes are excluded, here's why" detail the CLI always
      // prints. This mirrors that detail directly in the approval panel.
      if (payload.withheld_code_fixes && payload.withheld_code_fixes.length > 0) {
        body.innerHTML += `<div class="approval-row" style="color:var(--yellow); border-color: var(--yellow);">⚠ ${payload.withheld_code_fixes.length} fix(es) WITHHELD -- excluded by default</div>`;
        payload.withheld_code_fixes.forEach(f => {
          body.innerHTML += `<div class="approval-row"><span class="score-fail">${f.path}</span> (${f.findings_addressed} finding(s)) [critic ${f.critic_score}/10 ${f.critic_verdict}]</div>`;
          (f.problems || []).slice(0, 3).forEach(p => {
            body.innerHTML += `<div class="approval-row" style="padding-left:1rem; color:var(--text-dim);">• ${p}</div>`;
          });
        });
      }

      body.innerHTML += `<div class="approval-question">Open a real GitHub PR with these fixes?</div>`;
    }

    // Only the PR checkpoint has a meaningful "approve all, including
    // withheld" choice -- the report checkpoint is a plain proceed/don't.
    const approveAllBtn = document.getElementById('btn-approve-all');
    if (payload.checkpoint === 'pr_approval' && payload.withheld_code_fixes && payload.withheld_code_fixes.length > 0) {
      approveAllBtn.style.display = 'block';
      approveAllBtn.disabled = false;
    } else {
      approveAllBtn.style.display = 'none';
    }

    document.getElementById('btn-approve').disabled = false;
    document.getElementById('btn-reject').disabled = false;
    document.getElementById('btn-approve-all').disabled = false;
    panel.classList.add('visible');
  }

  function sendApproval(answer) {
    document.getElementById('btn-approve').disabled = true;
    document.getElementById('btn-reject').disabled = true;
    document.getElementById('btn-approve-all').disabled = true;
    document.getElementById('approval-body').innerHTML +=
      `<div class="approval-question" style="color:var(--text-dim);">Continuing scan...</div>`;

    fetch(`/approve/${currentScanId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answer })
    })
      .then(async (res) => {
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Request failed (${res.status})`);
        }
        document.getElementById('approval-panel').classList.remove('visible');
      })
      .catch((err) => {
        document.getElementById('approval-body').innerHTML +=
          `<div class="approval-question" style="color:var(--red, #ef4444);">⚠ ${err.message}</div>`;
        document.getElementById('btn-approve').disabled = false;
        document.getElementById('btn-reject').disabled = false;
        document.getElementById('btn-approve-all').disabled = false;
      });
  }

  function showResult(data) {
    document.getElementById('stat-code').textContent  = data.code_findings;
    document.getElementById('stat-dep').textContent   = data.dep_findings;
    document.getElementById('stat-total').textContent = data.total_findings;
    document.getElementById('btn-report').href = `/report/${data.scan_id}`;

    const existingNote = document.getElementById('withheld-note');
    if (existingNote) existingNote.remove();
    if (data.withheld_fixes_count > 0) {
      const note = document.createElement('div');
      note.id = 'withheld-note';
      note.style.cssText = 'font-family:var(--mono); font-size:11px; color:var(--yellow); margin-bottom:.75rem;';
      note.textContent = `⚠ ${data.withheld_fixes_count} fix(es) withheld -- see report for details`;
      document.getElementById('btn-report').insertAdjacentElement('beforebegin', note);
    }

    const btnPr = document.getElementById('btn-pr');
    if (data.pr_url) {
      btnPr.href = data.pr_url;
      btnPr.style.display = 'block';
    } else {
      btnPr.style.display = 'none';
    }

    // View Fixes & Diffs -- only worth showing if there's anything to
    // actually diff. code_fixes_count + withheld_fixes_count covers
    // both confirmed and withheld, since a withheld fix's diff is
    // often the more interesting one to inspect (why did this get
    // rejected?).
    const btnFixes = document.getElementById('btn-fixes');
    diffModalScanId = data.scan_id;
    if ((data.code_fixes_count || 0) + (data.withheld_fixes_count || 0) > 0) {
      btnFixes.style.display = 'block';
    } else {
      btnFixes.style.display = 'none';
    }

    document.getElementById('result-panel').classList.add('visible');
  }

  // ── Diff viewer modal ──────────────────────────────────────────────
  let diffModalScanId = null;

  function openDiffModal() {
    if (!diffModalScanId) return;
    const sidebar = document.getElementById('diff-modal-sidebar');
    const body = document.getElementById('diff-modal-body');
    sidebar.innerHTML = '<div class="diff-line diff-ctx">Loading…</div>';
    body.innerHTML = '<div class="diff-line diff-ctx">Select a file on the left to view its diff.</div>';
    document.getElementById('diff-modal-backdrop').classList.add('visible');

    fetch(`/scan/${diffModalScanId}/fixes`)
      .then(res => res.json())
      .then(data => {
        sidebar.innerHTML = '';
        (data.fixes || []).forEach(f => {
          const item = document.createElement('div');
          item.className = 'diff-file-item';
          item.dataset.idx = f.index;
          const badge = f.confirmed ? '✓' : '⚠';
          item.textContent = `${badge} ${f.path}`;
          item.onclick = () => selectDiffFile(f.index, item);
          sidebar.appendChild(item);
        });
        // Auto-select the first file so the modal isn't empty on open.
        const first = sidebar.querySelector('.diff-file-item');
        if (first) selectDiffFile(first.dataset.idx, first);
      })
      .catch(() => {
        sidebar.innerHTML = '<div class="diff-line diff-ctx">Failed to load fixes.</div>';
      });
  }

  function selectDiffFile(idx, itemEl) {
    document.querySelectorAll('.diff-file-item').forEach(el => el.classList.remove('active'));
    if (itemEl) itemEl.classList.add('active');

    const body = document.getElementById('diff-modal-body');
    body.innerHTML = '<div class="diff-line diff-ctx">Loading…</div>';

    fetch(`/scan/${diffModalScanId}/diff/${idx}`)
      .then(res => res.text())
      .then(html => { body.innerHTML = html; })
      .catch(() => {
        body.innerHTML = '<div class="diff-line diff-ctx">Failed to load diff.</div>';
      });
  }

  function closeDiffModal() {
    document.getElementById('diff-modal-backdrop').classList.remove('visible');
  }

  function showError(msg) {
    const box = document.getElementById('error-box');
    box.textContent = '✗ ' + msg;
    box.classList.add('visible');
    document.getElementById('btn-scan').disabled = false;
  }

  function resetUI() {
    document.getElementById('progress-panel').classList.remove('visible');
    document.getElementById('approval-panel').classList.remove('visible');
    document.getElementById('result-panel').classList.remove('visible');
    document.getElementById('error-box').classList.remove('visible');
    closeDiffModal();
    diffModalScanId = null;
    document.getElementById('btn-scan').disabled = false;
    document.getElementById('url-input').value = '';
    selectedFile = null;
    document.getElementById('file-name').textContent = 'zip your folder first · max 100MB';
    document.getElementById('pulse-dot').style.background = 'var(--accent)';
    document.getElementById('pulse-dot').style.animation = 'pulse 1.4s infinite';
    currentScanId = null;
  }

  // If we landed here via a "Re-scan" from the /repos page, the redirect
  // carries ?scan_id=<id> -- pick that up and resume watching it instead
  // of showing an empty "start a new scan" form.
  (function resumeFromQueryParam() {
    const params = new URLSearchParams(window.location.search);
    const scanId = params.get('scan_id');
    if (scanId) {
      currentScanId = scanId;
      document.getElementById('btn-scan').disabled = true;
      showProgress(scanId);
    }
  })();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("\n  🔍 Vulnerability Agent — Web UI")
    print("  ─────────────────────────────────")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000, threaded=True)