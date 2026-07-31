"""
tools/emailer.py
----------------
Sends a scan's HTML report to the repo owner's own Gmail account,
immediately once report_node finishes generating it (see the hook in
agent/nodes.py).

Design (see models.GmailConnection's docstring for the full rationale):
  - Each user separately "Connects Gmail" (web_app.py's /connect/gmail
    routes) -- a distinct, sensitive-scope consent from ordinary login,
    since it needs offline access (a refresh token) to send mail later
    without the user being present.
  - The refresh token is stored ENCRYPTED at rest (Fernet, symmetric --
    the app needs to read it back, so this isn't a password-hash
    situation; a proper KMS/secrets-manager is future work, noted in
    NEXT STEPS as a possible hardening item, but Fernet is a real
    improvement over plaintext for a student/portfolio deployment).
  - Sending uses a short-lived access token minted fresh from the
    refresh token on every send (never cached across calls) via
    google-auth's Request()/Credentials.refresh(), then the Gmail API's
    users.messages.send -- NOT smtplib, since Gmail's SMTP requires
    either a full password or an App Password, both of which this
    project deliberately avoids storing for users other than the app
    owner.

Failure handling: a failed send should NEVER fail the scan itself --
report generation already succeeded and the report is still reachable
at /report/<scan_id> regardless of whether the email went out. Callers
should catch EmailSendError and log, not propagate.
"""

import base64
import os
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from logging_config import get_logger
from tools.crypto_utils import encrypt_secret, decrypt_secret, SecretDecryptError

logger = get_logger(__name__)

GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


class EmailSendError(Exception):
    """Raised for any failure sending the report email. Callers (e.g.
    report_node) should catch this specifically and log a warning --
    never let it bubble up and fail the scan itself."""


def encrypt_refresh_token(raw_refresh_token: str) -> str:
    return encrypt_secret(raw_refresh_token)


def decrypt_refresh_token(encrypted_refresh_token: str) -> str:
    try:
        return decrypt_secret(encrypted_refresh_token)
    except SecretDecryptError:
        # Re-raised as EmailSendError (not SecretDecryptError) so
        # existing callers written against emailer.py's original
        # exception type keep working unchanged.
        raise EmailSendError(
            "Could not decrypt stored Gmail refresh token -- the encryption "
            "key may have changed. Ask the user to reconnect Gmail."
        )


def send_report_email(gmail_connection, subject: str, report_html: str) -> None:
    """gmail_connection: a models.GmailConnection row (needs
    .gmail_address and .encrypted_refresh_token).
    Raises EmailSendError on any failure -- callers must catch this and
    continue; a failed email must never fail the underlying scan.
    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise EmailSendError("GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET not configured -- cannot refresh Gmail token.")

    refresh_token = decrypt_refresh_token(gmail_connection.encrypted_refresh_token)

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=[GMAIL_SEND_SCOPE],
    )

    try:
        creds.refresh(Request())
    except Exception as e:
        # Refresh tokens can be revoked by the user (Google account
        # settings -> Third-party access) or expire from long disuse --
        # either way, this specific user needs to reconnect; it's not
        # an app-wide outage.
        raise EmailSendError(f"Failed to refresh Gmail access token for {gmail_connection.gmail_address}: {e}")

    message = MIMEText(report_html, "html")
    message["to"] = gmail_connection.gmail_address
    message["from"] = gmail_connection.gmail_address
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
    except HttpError as e:
        raise EmailSendError(f"Gmail API rejected the send for {gmail_connection.gmail_address}: {e}")
    except Exception as e:
        raise EmailSendError(f"Unexpected error sending Gmail report to {gmail_connection.gmail_address}: {e}")

    logger.info(f"Report emailed to {gmail_connection.gmail_address}")