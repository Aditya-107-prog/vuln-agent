"""
Tool 5: CVE Enricher
---------------------
For each dependency vulnerability found by pip-audit (Python),
govulncheck (Go), or osv-scanner (Java), this tool hits the OSV.dev
API to fetch:
  - CVSS severity score (0.0 - 10.0)
  - Severity label (Critical / High / Medium / Low)
  - Affected versions
  - Patch/fix information

For code findings from bandit/semgrep, we use the scanner's own
severity field since those aren't CVEs -- they're code pattern issues.

ECOSYSTEM (fixed this phase): previously "ecosystem": "PyPI" was
hardcoded into every OSV query, which was silently wrong for any
non-Python finding -- a Go or Maven package queried against PyPI
would just come back empty (OSV.dev's query is ecosystem-scoped), not
error, so this bug would have looked like "no CVEs found" rather than
an obvious failure. Now reads finding["ecosystem"], defaulting to
"PyPI" only for findings that don't specify one (i.e. every existing
pip-audit finding, so Python behavior is unchanged).

OSV.dev API: https://api.osv.dev/v1/query
No API key needed. Free. Run by Google.
"""

import requests
import time

from logging_config import get_logger

logger = get_logger(__name__)

try:
    from cvss import CVSS2, CVSS3, CVSS4
    _CVSS_LIB_AVAILABLE = True
except ImportError:
    _CVSS_LIB_AVAILABLE = False
    logger.warning(
        "[cve_enricher] 'cvss' package not installed -- CVSS vector strings "
        "from OSV.dev cannot be parsed into numeric scores, so severity "
        "will fall back to generic 'HIGH' for most findings. "
        "Run: pip install cvss"
    )


def _log_cvss_parse_failure(package: str, sev_type: str, vector: str, error: Exception):
    logger.warning(f"[cve_enricher] Failed to parse {sev_type} vector for {package} ({vector!r}): {error}")


OSV_API = "https://api.osv.dev/v1/query"

# Cap enrichment at top N dep findings to avoid timeouts on large repos
MAX_ENRICHED = 20


def severity_label(score: float) -> str:
    """Convert a CVSS numeric score to a human label."""
    if score >= 9.0:
        return "CRITICAL"
    elif score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    else:
        return "LOW"


def enrich_dep_finding(finding: dict) -> dict:
    """
    Query OSV.dev for a single vulnerable package and enrich the finding.
    Returns the finding with added severity/CVSS data.
    """
    package = finding["package"]
    version = finding["installed_version"]
    # Default "PyPI" preserves exact prior behavior for every existing
    # pip-audit finding, which never set this key. Go/Java findings
    # set "ecosystem" explicitly (see dep_scanner.py).
    ecosystem = finding.get("ecosystem", "PyPI")

    try:
        response = requests.post(
            OSV_API,
            json={
                "version": version,
                "package": {
                    "name": package,
                    "ecosystem": ecosystem
                }
            },
            timeout=10
        )

        if response.status_code != 200:
            # OSV didn't respond well — return finding as-is
            return {**finding, "cvss_score": None, "severity": "UNKNOWN", "osv_ids": []}

        data = response.json()
        vulns = data.get("vulns", [])

        if not vulns:
            return {**finding, "cvss_score": None, "severity": "UNKNOWN", "osv_ids": []}

        # Pull CVSS score from the first vuln that has one
        cvss_score = None
        osv_ids = []

        for vuln in vulns:
            osv_ids.append(vuln.get("id", ""))

            # OSV stores severity in different places depending on the
            # source database. GitHub Security Advisory-sourced
            # entries (the majority of real-world Maven/npm/PyPI
            # advisories) put CVSS in severity[] as a VECTOR STRING
            # (e.g. "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"),
            # not a plain number -- this has to be parsed to get a
            # base score. FIX: this used to have a dead `pass` here
            # (see prior version's comment "the actual numeric score
            # is separate") which meant EVERY finding relying on this
            # field silently fell through to the database_specific
            # fallback below, which most GHSA-sourced entries don't
            # populate -- so nearly everything defaulted to generic
            # "HIGH" regardless of true severity (a 9.8 RCE and a 4.1
            # minor issue looked identical in the report).
            if cvss_score is None and _CVSS_LIB_AVAILABLE:
                for severity_entry in vuln.get("severity", []):
                    sev_type = severity_entry.get("type", "")
                    vector = severity_entry.get("score", "")
                    if not vector:
                        continue
                    try:
                        # NOTE: cast to float() explicitly -- the cvss
                        # library's .base_score returns a
                        # decimal.Decimal, not a plain float. This bit
                        # us in the real run that first exercised this
                        # fix: report_generator.py later does
                        # json.dumps() on the enriched findings, and
                        # json.dumps() cannot serialize Decimal at all
                        # (raises TypeError), crashing the whole report
                        # step. float() here is required, not cosmetic.
                        if sev_type == "CVSS_V3" and vector.startswith("CVSS:3"):
                            cvss_score = float(CVSS3(vector).base_score)
                            break
                        elif sev_type == "CVSS_V4" and vector.startswith("CVSS:4"):
                            cvss_score = float(CVSS4(vector).base_score)
                            break
                        elif sev_type == "CVSS_V2":
                            # CVSS v2 vectors don't carry a "CVSS:x.y/"
                            # prefix the way v3/v4 do (e.g.
                            # "AV:N/AC:L/Au:N/C:P/I:P/A:P") -- the cvss
                            # library's CVSS2 class handles this
                            # format directly.
                            cvss_score = float(CVSS2(vector).base_score)
                            break
                    except Exception as e:
                        _log_cvss_parse_failure(package, sev_type, vector, e)
                        continue  # try the next severity entry rather than giving up on this vuln entirely

            # Fallback: some OSV entries (particularly OSV-native, non-
            # GHSA-sourced ones) DO populate database_specific with an
            # already-numeric score directly -- prefer the vector-based
            # parse above when available since it's more universally
            # present, but use this when severity[] was empty/unparseable.
            db = vuln.get("database_specific", {})
            score = db.get("cvss_score") or db.get("severity_score")
            if score and cvss_score is None:
                try:
                    cvss_score = float(score)
                except (ValueError, TypeError):
                    pass

        severity = severity_label(cvss_score) if cvss_score else "HIGH"
        # Default to HIGH if no score found — better to over-report than under

        logger.info(f"{package} {version} -> {severity} (CVSS: {cvss_score})")

        return {
            **finding,          # keep all original fields
            "cvss_score": cvss_score,
            "severity": severity,
            "osv_ids": osv_ids[:3]  # cap at 3 IDs
        }

    except requests.Timeout:
        logger.warning(f"Timeout on {package} -- skipping enrichment")
        return {**finding, "cvss_score": None, "severity": "HIGH", "osv_ids": []}

    except Exception as e:
        logger.error(f"Error enriching {package}: {e}")
        return {**finding, "cvss_score": None, "severity": "HIGH", "osv_ids": []}


def enrich_code_finding(finding: dict) -> dict:
    """
    Code findings from bandit already have severity (HIGH/MEDIUM/LOW).
    We just standardise the format to match dep findings.
    """
    return {
        **finding,
        "cvss_score": None,     # bandit doesn't give CVSS scores
        "severity": finding.get("severity", "MEDIUM"),
        "osv_ids": []
    }


def run_cve_enrichment(code_findings: list, dep_findings: list) -> list:
    """
    Enrich all findings and return a unified sorted list.

    Args:
        code_findings: list from bandit (via code_scan_node)
        dep_findings:  list from pip-audit (via dep_scan_node)

    Returns:
        Single sorted list of all enriched findings, highest severity first
    """
    logger.info(f"Enriching {len(dep_findings)} dep findings (capped at {MAX_ENRICHED})")
    logger.info(f"Passing through {len(code_findings)} code findings")

    enriched = []

    # Enrich dependency findings via OSV.dev
    for finding in dep_findings[:MAX_ENRICHED]:
        enriched_finding = enrich_dep_finding(finding)
        enriched_finding["finding_type"] = "dependency"
        enriched.append(enriched_finding)
        time.sleep(0.2)  # be a polite API citizen — small delay between calls

    # Standardise code findings (no API call needed)
    for finding in code_findings:
        enriched_finding = enrich_code_finding(finding)
        enriched_finding["finding_type"] = "code"
        enriched.append(enriched_finding)

    # Sort everything: CRITICAL → HIGH → MEDIUM → LOW → UNKNOWN
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
    enriched.sort(key=lambda x: severity_order.get(x.get("severity", "UNKNOWN"), 4))

    logger.info(f"{len(enriched)} total enriched findings")
    return enriched