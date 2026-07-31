"""
One-off cleanup script: marks any Scan row stuck at status='running' as
'error', with a clear note explaining why. Run this once, then delete it --
it's not part of the app itself.

Usage (from the vuln-agent project root, with DATABASE_URL set to your
REAL dev DB, not vuln_agent_test):

    python cleanup_stale_scans.py
"""

from web_app import app
from models import db, Scan
from datetime import datetime, timezone

app.app_context().push()

stale = Scan.query.filter_by(status="running").all()
print(f"Found {len(stale)} scan(s) stuck at running:")
for s in stale:
    print(f"  {s.id}  repo_id={s.repo_id}  created_at={s.created_at}")
    s.status = "error"
    s.error = "Orphaned: scan thread died (app restart) without updating status."
    s.completed_at = datetime.now(timezone.utc)

db.session.commit()
print("Marked as error.")
