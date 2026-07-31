"""
logging_config.py
------------------
Central logging setup for vuln-agent. Replaces the scattered print()
calls that used to be the only record of what a scan did -- once a
terminal closed or scrolled past, that history was gone (this is
exactly what happened with a stuck scan during manual testing: nothing
survived to explain what happened after the fact).

Usage, in any module:

    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("scan started")
    logger.error("bandit failed: %s", err)

Every log line automatically includes whichever scan_id is currently
bound to the executing thread (see bind_scan_id / clear_scan_id below),
so `grep <scan_id> logs/vuln_agent.log` reconstructs the full timeline
of one scan -- across web_app.py, agent/nodes.py, and every tools/
module -- without having to thread scan_id through every function
signature by hand.

Design notes:
- File handler: rotating, always captures INFO and above, regardless
  of AGENT_QUIET. The whole point of a log file is that it doesn't
  depend on someone having --verbose on at the time.
- Console handler: honors AGENT_QUIET (same env var the CLI already
  used to silence routine progress prints) -- WARNING+ only when quiet,
  INFO+ otherwise. This preserves today's CLI UX while adding a
  permanent record underneath it.
- scan_id binding uses contextvars.ContextVar, not a plain
  threading.local or module-level variable. This matters specifically
  because tools/fix_generator.py runs per-file fix generation across a
  ThreadPoolExecutor, and (for the same reason it already had to do
  this for Langfuse span context) captures a contextvars.copy_context()
  snapshot per submitted worker. threading.local() is NEVER propagated
  into a new thread by anything -- a ContextVar captured via
  copy_context().run() is, automatically, for free, with zero changes
  needed in fix_generator.py's executor code. Binding still happens
  once per logical scan (in web_app.py's background thread, or the CLI)
  and rides along through every subsequent copy_context().run() call
  downstream.
"""

import logging
import logging.handlers
import os
import contextvars

LOG_DIR = os.environ.get("AGENT_LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "vuln_agent.log")

_configured = False
_scan_id_var = contextvars.ContextVar("scan_id", default=None)


def bind_scan_id(scan_id):
    """Call at the start of a scan (CLI main(), or the background
    thread in web_app.py) so every log line emitted for the rest of
    the scan is automatically tagged with it -- including inside any
    ThreadPoolExecutor worker that's launched via
    contextvars.copy_context().run(), since that copies whatever
    ContextVar values were set at the point of the copy."""
    _scan_id_var.set(scan_id)


def clear_scan_id():
    """Call when a scan finishes (success or error) so a thread that
    gets reused (e.g. a thread pool) doesn't leak a stale scan_id into
    unrelated log lines later."""
    _scan_id_var.set(None)


class _ScanIdFilter(logging.Filter):
    """Injects the current context's scan_id into every LogRecord so the
    formatter can include it, defaulting to '-' outside of any scan
    (e.g. Flask request-handling code that isn't inside the scan
    context) instead of raising AttributeError on formatting."""

    def filter(self, record):
        record.scan_id = _scan_id_var.get() or "-"
        return True


def configure_logging():
    """Idempotent -- safe to call from multiple entry points (web_app.py
    AND agent.py both import tools/, and either could be the first to
    import logging_config) without double-registering handlers, which
    would otherwise duplicate every log line."""
    global _configured
    if _configured:
        return
    _configured = True

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger("vuln_agent")
    root.setLevel(logging.DEBUG)
    root.propagate = False  # don't also hand records to Flask's/root's own handlers

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s [%(scan_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    scan_id_filter = _ScanIdFilter()

    # Rotating file handler -- always INFO+, regardless of AGENT_QUIET.
    # 5MB per file, keep 5 backups, so this can't silently grow forever
    # on a long-lived dev machine.
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(scan_id_filter)
    root.addHandler(file_handler)

    # Console handler -- same AGENT_QUIET behavior the CLI already had:
    # WARNING+ only when quiet, INFO+ (routine progress) otherwise.
    console_handler = logging.StreamHandler()
    console_level = logging.WARNING if os.environ.get("AGENT_QUIET") == "1" else logging.INFO
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    console_handler.addFilter(scan_id_filter)
    root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """The only function most modules need. Configures logging on first
    call (see configure_logging's idempotency note above), then returns
    a logger namespaced under 'vuln_agent.<module>' so log lines are
    still traceable to their source module in the file/console output."""
    configure_logging()
    return logging.getLogger(f"vuln_agent.{name}")