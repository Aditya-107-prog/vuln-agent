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

## Limitations

- Python repositories only (no web/network scanning)
- Public GitHub repos only (no auth for private repos)
- Dependency scanning requires a requirements.txt / pyproject.toml / Pipfile