"""
tools/slack_notifier.py
------------------------
Posts a scan-completion summary to the repo owner's connected Slack
channel via an Incoming Webhook. Fires once the ENTIRE scan finishes
(findings + fix generation + PR, if any) -- deliberately later than
Gmail's auto-send, which fires as soon as the report itself exists.
That's a real behavioral difference, not an inconsistency: the report
email is "here's what we found," useful the moment it exists, while
the Slack ping is meant to represent "the whole run is done, here's
the final outcome" for a channel a team might be watching together.

Failure handling matches emailer.py's principle exactly: a failed
Slack post must never fail the scan itself. The scan already
succeeded (or failed) on its own terms; this is a best-effort notify
on top, not a required step.
"""

import requests

from logging_config import get_logger
from tools.crypto_utils import encrypt_secret, decrypt_secret, SecretDecryptError

logger = get_logger(__name__)


class SlackSendError(Exception):
    """Raised for any failure posting to Slack. Callers must catch this
    and log a warning, never let it propagate and fail the scan."""


def encrypt_webhook_url(raw_url: str) -> str:
    return encrypt_secret(raw_url)


def decrypt_webhook_url(encrypted_url: str) -> str:
    try:
        return decrypt_secret(encrypted_url)
    except SecretDecryptError:
        raise SlackSendError(
            "Could not decrypt stored Slack webhook URL -- the encryption "
            "key may have changed. Ask the user to reconnect Slack."
        )


def send_scan_summary(
    encrypted_webhook_url: str,
    repo_name: str,
    scan_id: str,
    code_findings: int,
    dep_findings: int,
    fixes_confirmed: int,
    fixes_withheld: int,
    pr_url: str = None,
    report_url: str = None,
) -> None:
    """Raises SlackSendError on any failure -- callers must catch and
    continue; a failed Slack post must never fail the underlying scan."""
    webhook_url = decrypt_webhook_url(encrypted_webhook_url)

    lines = [
        f"*Scan complete: `{repo_name}`*",
        f"Findings: {code_findings} code, {dep_findings} dependency",
        f"Fixes: {fixes_confirmed} confirmed, {fixes_withheld} withheld",
    ]
    if pr_url:
        lines.append(f"Pull request: {pr_url}")
    if report_url:
        lines.append(f"Full report: {report_url}")

    # Slack's Incoming Webhooks accept either a plain "text" field or
    # richer Block Kit markup -- plain text is deliberately used here
    # rather than blocks: it's readable in every Slack client (including
    # notification previews) without needing to reason about block
    # rendering edge cases, and this message has no interactive elements
    # that would justify the extra complexity.
    payload = {"text": "\n".join(lines)}

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
    except requests.RequestException as e:
        raise SlackSendError(f"Network error posting to Slack for scan {scan_id}: {e}")

    if response.status_code != 200:
        # Slack's webhook errors are plain text in the body ("invalid_payload",
        # "channel_not_found", etc.), not JSON -- surfaced directly since
        # it's already human-readable and more useful than a generic
        # "request failed" message.
        raise SlackSendError(
            f"Slack rejected the webhook post for scan {scan_id}: "
            f"HTTP {response.status_code} - {response.text[:200]}"
        )

    logger.info(f"Scan summary posted to Slack for scan {scan_id}")