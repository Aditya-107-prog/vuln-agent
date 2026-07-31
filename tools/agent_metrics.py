"""
tools/agent_metrics.py
----------------------
Prometheus metrics specifically about HOW WELL the agent's fix-generation
pipeline is performing -- as opposed to web_app.py's metrics (scan counts,
durations, HTTP-level stuff). Split into its own module, importable from
critic.py/fix_generator.py, because those files are also used by the CLI
path (agent.py), which has no Flask app at all -- importing anything from
web_app.py there would break CLI usage the same way putting DB/email
logic directly in agent/nodes.py would have (see the Gmail auto-send
design notes in web_app.py's _maybe_email_report).

prometheus_client's Counter/Histogram objects register themselves into
the SAME global default registry regardless of which module defines
them -- so as long as this module gets imported at least once during a
scan (it always will, since critic.py/fix_generator.py import it),
these metrics show up in web_app.py's existing /metrics endpoint
automatically. No wiring needed on the web_app.py side.
"""

from prometheus_client import Counter, Histogram

# One observation per critic call that actually returned a numeric
# score (i.e. CRITIC_ENABLED and Bedrock didn't error out). Buckets
# chosen to make "how often do fixes pass on the first try" answerable
# directly: bucket boundaries sit on the actual score values a fix can
# receive, not evenly-spaced arbitrary numbers.
CRITIC_SCORE = Histogram(
    "vuln_agent_critic_score",
    "Critic score (0-10) per fix attempt that received a real verdict",
    buckets=[0, 2, 4, 6, 7, 8, 9, 10],
)

# One observation per FILE (not per attempt) once its retry loop ends
# -- value is however many attempts (1-3) that file actually used
# before either passing or exhausting retries. This is what answers
# "is the agent needing more retries lately", a leading indicator of
# either a harder batch of findings or a degrading model/prompt.
FIX_ATTEMPTS_PER_FILE = Histogram(
    "vuln_agent_fix_attempts_per_file",
    "Number of LLM attempts used before a file's fix was accepted or given up on",
    buckets=[1, 2, 3],
)

# One increment per FILE's FINAL verdict (after all retries exhausted
# or an early pass) -- this is "fix success rate" in its most direct
# form: confirmed_total / (confirmed_total + everything_else_total).
FIX_FINAL_OUTCOME = Counter(
    "vuln_agent_fix_final_outcome_total",
    "Final verdict per file after its retry loop ends",
    ["verdict"],  # "pass" | "needs_improvement" | "fail" | "unavailable"
)