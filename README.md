# Vulnerability Finder Agent

A LangGraph-powered AI security agent that scans Python repositories for vulnerabilities and produces styled HTML reports.

Point it at a GitHub URL or local folder — it clones/reads the repo, runs static analysis and dependency checks, enriches findings with real CVE severity data, and uses an LLM to write a professional security report.

## Stack

- **LangGraph** — agent graph framework (state, nodes, conditional edges)
- **Groq / Llama 3** — LLM for report generation (free tier)
- **bandit** — Python SAST scanner (~100 security patterns)
- **pip-audit** — dependency vulnerability checker
- **OSV.dev API** — CVE severity data, no key needed

Everything is free. No credit card required.

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env
# Edit .env and paste your key from https://console.groq.com/keys

# 4. Verify setup
python hello_graph.py
```

## Usage

```bash
# Scan a public GitHub repo
python agent.py --target https://github.com/we45/Vulnerable-Flask-App

# Scan a local folder
python agent.py --target /path/to/your/project

# Scan current directory, save report elsewhere
python agent.py --target . --output ./my-reports

# Show detailed tool output
python agent.py --target https://github.com/user/repo --verbose
```

The HTML report opens in any browser. Typical run time: 30–90 seconds.

## Project structure

```
vuln-agent/
├── agent/
│   ├── state.py        # VulnScanState TypedDict
│   ├── nodes.py        # All 6 graph nodes
│   └── graph.py        # Graph wiring + compilation
├── tools/
│   ├── github_fetcher.py   # Clones public GitHub repos
│   ├── local_reader.py     # Reads local directories
│   ├── code_scanner.py     # Wraps bandit
│   ├── dep_scanner.py      # Wraps pip-audit
│   ├── cve_enricher.py     # Hits OSV.dev API
│   └── report_generator.py # Calls Groq + writes HTML
├── reports/            # Generated HTML reports land here
├── agent.py            # CLI entry point
├── hello_graph.py      # Day 1 warm-up
└── requirements.txt
```

## How it works

```
Input (GitHub URL or local path)
        ↓
    Router node  ──→  github_fetch  ─┐
                  └→  local_read   ──┤
                                     ↓
                               code_scan  (bandit)
                                     ↓
                               dep_scan   (pip-audit)
                                     ↓
                               cve_enrich (OSV.dev)
                                     ↓
                               report     (Groq LLM → HTML)
```

## Observability

### Local (Prometheus + Grafana)

```bash
docker compose up -d
```

Prometheus `localhost:9090`, Grafana `localhost:3000` (`admin`/`admin`). The
`vuln-agent` dashboard and datasource are pre-provisioned. `web_app.py` must
already be running on port 5000 — compose only scrapes it, it does not run it.

`/metrics` is gated by a bearer token. Set `METRICS_TOKEN` in `.env` to the
same literal value in `observability/prometheus.yml`, or leave it unset to
run the endpoint open locally.

### Production (Railway → Grafana Cloud)

Railway ignores `docker-compose.yml`, so production uses a second Railway
service running [Grafana Alloy](https://grafana.com/docs/alloy/), which
scrapes `/metrics` over Railway's **private** network and `remote_write`s to
Grafana Cloud. `/metrics` is never exposed publicly, and metrics keep
flowing whether or not your laptop is on.

```
vuln-agent (web)  ──private net──>  alloy  ──remote_write──>  Grafana Cloud
   /metrics                        scrape                       dashboards
   + METRICS_TOKEN                 30s                          + alerts
```

**1. Generate a token**

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set it as `METRICS_TOKEN` on **both** Railway services. Mismatched values
make every scrape return 404 — that's the intended failure mode (a 401 would
confirm the endpoint exists), so it fails silently. Check
`up{job="vuln-agent"}` in Grafana if no data appears.

**2. Get Grafana Cloud credentials**

Free tier is enough (10k series; this app emits well under 100). In Grafana
Cloud → your stack → Prometheus → **Send Metrics**, copy the remote-write
URL, the numeric username, and create an access-policy token scoped to
`metrics:write`.

**3. Create the Alloy service**

In the **same Railway project** as vuln-agent (same project = same private
network):

- New service → same repo
- Settings → **Root Directory** = `observability/alloy`
  (skip this and Railway builds the repo-root Dockerfile — a second copy of
  the web app)
- Do **not** assign a public domain. Alloy only needs outbound access.
- Variables:

  | Variable | Value |
  |---|---|
  | `METRICS_TOKEN` | same value as the web service |
  | `GRAFANA_CLOUD_PROM_URL` | `https://prometheus-prod-NN-REGION.grafana.net/api/prom/push` |
  | `GRAFANA_CLOUD_PROM_USER` | numeric instance ID |
  | `GRAFANA_CLOUD_PROM_PASSWORD` | `glc_...` access policy token |
  | `VULN_AGENT_TARGET` | *(optional)* `vuln-agent.railway.internal:$PORT` |

**4. Confirm the target port**

`VULN_AGENT_TARGET` defaults to `vuln-agent.railway.internal:8080`. Railway
assigns `$PORT` per service and it is **not** 5000, so check the web
service's `PORT` variable and override if it differs. The hostname is the
Railway **service** name.

**5. Verify**

In Grafana Cloud → Explore:

```promql
up{job="vuln-agent"}                # 1 = scrape succeeding
vuln_agent_scans_total              # app counters arriving
up{job="alloy"}                     # Alloy itself is alive
```

If `up{job="alloy"}` reports but `up{job="vuln-agent"}` is absent, Alloy is
running and the scrape is failing — almost always a wrong `$PORT` in
`VULN_AGENT_TARGET` or a mismatched `METRICS_TOKEN`.

**Notes**

- Alloy runs in agent mode with its WAL in `/tmp` — no Railway volume
  needed, since Grafana Cloud holds the durable copy. A redeploy loses at
  most a few minutes of buffered samples.
- The app binds `[::]` rather than `0.0.0.0` because Railway's private
  network is IPv6-only. An IPv4-only listener is unreachable at
  `*.railway.internal`.
- `/metrics` re-queries Postgres on every scrape
  (`tools/db_metrics_collector.py`), which is why the endpoint is
  authenticated rather than left open — and why `scrape_interval` is 30s
  rather than something aggressive.

## Limitations

- Python repositories only (no web/network scanning)
- Public GitHub repos only (no auth for private repos)
- Dependency scanning requires a requirements.txt / pyproject.toml / Pipfile