"""
tools/db_metrics_collector.py
------------------------------
A Prometheus custom Collector that reports scan history metrics
computed LIVE from the Postgres `scans` table on every /metrics
scrape, instead of from in-memory prometheus_client Counters.

Why this exists: the Counter/Histogram objects in web_app.py
(SCANS_TOTAL, CODE_FINDINGS_TOTAL, etc.) live in the Python process's
memory. Every time web_app.py restarts -- a crash, a redeploy, a
manual restart during development -- those counters reset to 0,
because nothing outside the process was ever tracking them. The
`scans` table in Postgres, on the other hand, already durably records
every scan that ever ran, independent of how many times the web
process has restarted since.

This collector re-derives the metrics from that table at scrape time
(one small aggregate query per metric family), so what Prometheus
sees is always "the true historical total right now" -- restart-proof
by construction, since there's no running total being kept in memory
to lose in the first place.

Metric names intentionally use a "_db" suffix (e.g.
vuln_agent_scans_total_db) to avoid colliding with the existing
in-memory metrics of the same shape in web_app.py. Both can coexist:
the in-memory ones are fine for "did the LAST few minutes look
healthy" (extremely cheap, no DB round-trip), while these are the
ones to point Grafana panels at for "trend across every run we've
ever had, including before the last restart."
"""

from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector

from logging_config import get_logger

logger = get_logger(__name__)


class DBBackedCollector(Collector):
    """Registered once, at app startup, against Prometheus's default
    REGISTRY. Prometheus calls .collect() itself on every scrape --
    nothing else needs to call this."""

    def __init__(self, app):
        # Needs the Flask app to open a DB session at collect()-time,
        # since collect() runs outside of any Flask request context
        # (it's invoked by prometheus_client machinery when /metrics
        # is scraped, not from inside a route handler).
        self.app = app

    def collect(self):
        # Import here, not at module load time -- avoids a circular
        # import with models.py/web_app.py, and means a broken DB
        # connection only affects this collector's yield, not the
        # whole module import.
        from sqlalchemy import func
        from models import db, Scan

        with self.app.app_context():
            try:
                scans_family = CounterMetricFamily(
                    "vuln_agent_scans_total_db",
                    "Total scans, by final status and trigger (recomputed from Postgres every scrape -- survives app restarts)",
                    labels=["status", "triggered_by"],
                )
                rows = (
                    db.session.query(Scan.status, Scan.triggered_by, func.count(Scan.id))
                    .group_by(Scan.status, Scan.triggered_by)
                    .all()
                )
                for status, triggered_by, count in rows:
                    scans_family.add_metric([status or "unknown", triggered_by or "unknown"], count)
                yield scans_family

                code_sum = db.session.query(func.coalesce(func.sum(Scan.code_findings_count), 0)).scalar()
                dep_sum = db.session.query(func.coalesce(func.sum(Scan.dep_findings_count), 0)).scalar()

                code_family = CounterMetricFamily(
                    "vuln_agent_code_findings_total_db",
                    "Sum of code findings across every scan ever run (recomputed from Postgres every scrape)",
                )
                code_family.add_metric([], code_sum)
                yield code_family

                dep_family = CounterMetricFamily(
                    "vuln_agent_dep_findings_total_db",
                    "Sum of dependency findings across every scan ever run (recomputed from Postgres every scrape)",
                )
                dep_family.add_metric([], dep_sum)
                yield dep_family

                confirmed_sum = db.session.query(func.coalesce(func.sum(Scan.code_fixes_count), 0)).scalar()
                withheld_sum = db.session.query(func.coalesce(func.sum(Scan.withheld_fixes_count), 0)).scalar()

                fixes_family = CounterMetricFamily(
                    "vuln_agent_fixes_total_db",
                    "AI-generated fixes, by outcome, across every scan ever run (recomputed from Postgres every scrape)",
                    labels=["outcome"],
                )
                fixes_family.add_metric(["confirmed"], confirmed_sum)
                fixes_family.add_metric(["withheld"], withheld_sum)
                yield fixes_family

                # A gauge, not a counter -- "how many scans exist right
                # now that never finished" is a current state, not a
                # cumulative total, so it can legitimately go down (e.g.
                # a stuck scan later gets marked errored).
                in_progress = (
                    db.session.query(func.count(Scan.id))
                    .filter(Scan.status.in_(["running", "waiting_approval"]))
                    .scalar()
                )
                in_progress_family = GaugeMetricFamily(
                    "vuln_agent_scans_in_progress_db",
                    "Scans currently running or awaiting approval (recomputed from Postgres every scrape)",
                )
                in_progress_family.add_metric([], in_progress)
                yield in_progress_family

            except Exception as e:
                # A scrape must never 500 just because this one
                # collector's DB query failed (e.g. transient DB
                # hiccup) -- log and yield nothing this round rather
                # than raising, so the rest of /metrics (the existing
                # in-memory counters) still comes back fine.
                logger.warning(f"[db_metrics_collector] failed to compute DB-backed metrics: {e}")
                return