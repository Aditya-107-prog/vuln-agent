FROM python:3.12-slim

# --- system deps + Go (for osv-scanner) ---------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# osv-scanner (Java dependency scanning) -- Linux amd64 static binary
RUN curl -L -o /usr/local/bin/osv-scanner \
        https://github.com/google/osv-scanner/releases/latest/download/osv-scanner_linux_amd64 \
    && chmod +x /usr/local/bin/osv-scanner

WORKDIR /app

# --- python deps ----------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir bandit pip-audit

# --- app code ---------------------------------------------------------------
COPY . .
# Windows binary isn't usable in this image; the Linux one above replaces it.
RUN rm -f osv-scanner.exe

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

# IMPORTANT: single worker only. web_app.py starts an in-process
# APScheduler background thread (scheduled repo re-scans) on import.
# Running >1 gunicorn worker would start that scheduler multiple times
# and fire duplicate scans. Scale via threads, not workers, unless the
# scheduler is externalized first.
#
# Shell form (not exec form) so ${PORT} expands -- platforms like Railway
# assign the port at runtime. Falls back to 5000 for local/compose runs.
#
# Bind [::] rather than 0.0.0.0: Railway's private network is IPv6-only,
# so an IPv4-only listener is unreachable at vuln-agent.railway.internal
# and the Alloy metrics scraper (observability/alloy/) gets connection
# refused. On Linux a [::] listener is dual-stack, so the public IPv4
# edge and local compose runs keep working -- this is strictly more
# permissive than 0.0.0.0, not a swap.
CMD gunicorn --bind [::]:${PORT:-5000} --workers 1 --threads 4 --timeout 120 web_app:app
