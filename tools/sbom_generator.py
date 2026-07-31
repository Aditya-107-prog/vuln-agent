"""
tools/sbom_generator.py
------------------------
Generates a CycloneDX 1.5 JSON SBOM (Software Bill of Materials) from
a scan's dependency findings.

SCOPE, READ THIS FIRST: this is a VULNERABILITY-ONLY SBOM -- it lists
only the packages pip-audit/govulncheck/osv-scanner flagged as having a
known vulnerability, not the repo's full dependency tree. A complete
SBOM (every installed package, vulnerable or not) would need to parse
requirements.txt/go.mod/pom.xml directly for the full list, which is a
separate piece of work (see tools/dep_scanner.py-type code) not yet
built. This is a real, deliberate scope limitation -- explicitly
chosen over guessing at manifest-parsing logic this session hasn't
seen, and flagged in the UI wherever this file is offered for download
so nobody mistakes it for a complete inventory.

FIELD NAMES: verified directly against tools/dep_scanner.py. Raw findings
carry "package", "installed_version", "fix_version", and a "vulns":
[{"id", "description"}] list. "ecosystem" is only set on Go/Maven
findings ("Go"/"Maven") -- the Python/pip-audit path never sets it,
so its absence specifically means Python, not "generic"/"unknown".
"""

import uuid
from datetime import datetime, timezone


def _get_first(d: dict, *keys, default=None):
    """Returns the first present, non-None value among several
    candidate key names -- see the FIELD NAME CAVEAT above for why
    this defensiveness exists instead of a single hardcoded key."""
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _purl_for(ecosystem: str, name: str, version: str) -> str:
    """Builds a Package URL (purl) -- CycloneDX's standard way to
    identify a specific package+version+ecosystem. Ecosystem names are
    normalized to purl's own vocabulary (pypi/golang/maven), which
    don't always match this project's internal naming."""
    eco_map = {"python": "pypi", "pip": "pypi", "pypi": "pypi", "go": "golang", "golang": "golang", "java": "maven", "maven": "maven"}
    purl_type = eco_map.get((ecosystem or "").lower(), "generic")
    return f"pkg:{purl_type}/{name}@{version}"


def generate_cyclonedx_sbom(dependency_findings: list, repo_name: str) -> dict:
    """dependency_findings: enriched findings with finding_type == "dependency"
    (i.e. final_state["enriched_findings"] filtered to that type -- these
    carry cvss_score/severity/osv_ids from cve_enrich_node in addition to
    whatever raw package/version fields the underlying scanner produced).

    Returns a CycloneDX 1.5-shaped dict, ready for json.dumps(). Every
    component gets a "vulnerabilities" entry -- unlike a normal SBOM
    where most components have none, EVERY component here has at least
    one, since only vulnerable packages are included at all (see the
    module docstring's SCOPE note).
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    components = []
    vulnerabilities = []

    for i, finding in enumerate(dependency_findings):
        name = _get_first(finding, "package", "package_name", "name", "dependency", default="unknown")
        version = _get_first(finding, "installed_version", "version", "current_version", default="unknown")
        # dep_scanner.py only tags "ecosystem" on Go/Maven findings --
        # the Python/pip-audit path never sets this key at all, so its
        # absence specifically (not a missing/unknown value) means
        # Python, not "generic". Confirmed against the actual
        # dep_scanner.py findings dicts, not guessed.
        ecosystem = _get_first(finding, "ecosystem", "language", "package_manager", default="Python")
        bom_ref = f"component-{i}-{name}"

        components.append({
            "type": "library",
            "bom-ref": bom_ref,
            "name": name,
            "version": version,
            "purl": _purl_for(ecosystem, name, version),
        })

        # dep_scanner.py's raw findings carry vulnerability data as a
        # "vulns": [{"id":..., "description":...}] list, not flat
        # cve_id/osv_ids fields -- those (osv_ids, cvss_score, severity)
        # only get added later by cve_enrich_node, and only apply to
        # the finding as a whole rather than per-individual-vuln, so
        # for MULTIPLE vulns on one package the per-vuln description
        # still needs to come from the original "vulns" list.
        raw_vulns = finding.get("vulns") or []
        osv_ids = finding.get("osv_ids") or []

        if raw_vulns:
            for v in raw_vulns:
                vulnerabilities.append({
                    "id": v.get("id", f"UNKNOWN-{i}"),
                    "affects": [{"ref": bom_ref}],
                    "ratings": [{
                        "score": finding.get("cvss_score"),
                        "severity": (finding.get("severity") or "unknown").lower(),
                        "method": "CVSSv3" if finding.get("cvss_score") is not None else "other",
                    }],
                    "description": v.get("description", ""),
                })
        else:
            # Fallback for a finding shape without "vulns" at all (e.g.
            # already-enriched-only data with just osv_ids) -- kept as a
            # safety net, not the primary path.
            vuln_ids = osv_ids or [_get_first(finding, "cve_id", "vulnerability_id", default=f"UNKNOWN-{i}")]
            for vuln_id in vuln_ids:
                vulnerabilities.append({
                    "id": vuln_id,
                    "affects": [{"ref": bom_ref}],
                    "ratings": [{
                        "score": finding.get("cvss_score"),
                        "severity": (finding.get("severity") or "unknown").lower(),
                        "method": "CVSSv3" if finding.get("cvss_score") is not None else "other",
                    }],
                    "description": finding.get("description") or finding.get("summary") or "",
                })

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "component": {"type": "application", "name": repo_name},
            "tools": [{"vendor": "vuln-agent", "name": "vuln-agent-sbom-generator", "version": "1.0"}],
        },
        # This note is embedded IN the file itself, not just in the UI
        # around it, so the scope limitation travels with the SBOM even
        # if someone downloads it and opens it somewhere else entirely.
        "properties": [
            {"name": "vuln-agent:sbom-scope", "value": "vulnerability-only -- not a complete dependency inventory"}
        ],
        "components": components,
        "vulnerabilities": vulnerabilities,
    }