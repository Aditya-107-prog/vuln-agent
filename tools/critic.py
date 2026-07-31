"""
tools/critic.py
----------------
A SECOND, independent model via AWS Bedrock that critiques the
Groq-generated fixes. Which model is configurable via BEDROCK_MODEL_ID
(e.g. Claude, Mistral Magistral) -- using a different model FAMILY than
whatever generated the fix (Groq Llama 3.3) is the deliberate part, not
which specific Bedrock model you point this at. That avoids the fix
generator grading its own homework, which is a well-known weakness of
same-model self-eval.

Produces a structured verdict (score 0-10, specific problems with
severity tags, suggestions) for a proposed code fix. fix_generator.py
uses this to decide whether to accept a fix or retry once with the
critic's feedback folded back in.

SEVERITY-AWARE VERDICT (added after a real incident):
Previously we derived "pass"/"fail" purely from the numeric score,
ignoring the actual content of "problems". This let a fix get scored
8/10 "pass" while the critic's own problem list said, in plain text,
that a JWT verification bypass was still present and unaddressed --
because nothing cross-checked the score against what was actually
written in problems. That fix got merged into a real PR.

Now each problem carries a severity tag ("critical"/"moderate"/"minor"),
and ANY problem tagged "critical" forces verdict="fail" regardless of
score. A high score can no longer override a critical problem the
critic itself identified.

LANGFUSE TRACING (added this phase):
Previously critique_fix() was completely invisible in LangFuse -- only
Groq calls (report generation, code fix generation) were traced. This
meant zero visibility into critic behavior: no record of what prompt
was sent, what Bedrock actually returned, how long it took, or how
many tokens it used, in either the pass or the failure path.

critique_fix() is now wrapped in a "generation"-type LangFuse
observation (bedrock-critic), nested under whatever parent trace is
active (generate_all_fixes -> ... -> critique_fix). Bedrock's usage
shape is different from Groq's -- Groq gives prompt_tokens/
completion_tokens/total_tokens as attributes on a usage object;
Bedrock's converse() gives inputTokens/outputTokens/totalTokens as
dict keys nested under response["usage"] -- so this needed its own
mapping rather than reusing the Groq usage_details logic verbatim.

The observation is updated with output + usage on the success path,
and with an error-tagged output on the exception path, so failed
critic calls are now visible in LangFuse too instead of vanishing
silently into just a console [ERROR] line.
"""

import os
import json
import boto3
from langfuse import get_client
from tools.agent_metrics import CRITIC_SCORE

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Default here is just a fallback for an unset env var -- the actual
# model in use is whatever BEDROCK_MODEL_ID is set to (e.g. Mistral
# Magistral). Nothing below this line assumes a specific model family;
# reasoning-model output ([THINK]...[/THINK] wrapping, see critique_fix)
# is already handled generically for exactly this reason.
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

CRITIC_ENABLED = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
LANGFUSE_ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

# Score thresholds, now used ONLY when no problem is tagged "critical".
# A critical-tagged problem always forces "fail", no matter the score.
PASS_THRESHOLD = 8       # score >= this -> pass (if no critical problems)
FAIL_THRESHOLD = 5       # score < this -> fail, else needs_improvement

VALID_SEVERITIES = {"critical", "moderate", "minor"}


def _normalize_problems(raw_problems) -> list:
    """Coerces the critic's problems list into the expected
    [{"text": str, "severity": str}, ...] shape. Handles the case where
    an older/unexpected response gives plain strings instead of dicts --
    those are treated as "moderate" severity (a safe middle default,
    since we can't know if they were meant to be critical) rather than
    silently dropped or crashing."""
    normalized = []
    for p in raw_problems or []:
        if isinstance(p, dict) and "text" in p:
            severity = str(p.get("severity", "")).lower()
            if severity not in VALID_SEVERITIES:
                _log(f"[critic] Unrecognized severity {severity!r} on problem, defaulting to 'moderate'")
                severity = "moderate"
            normalized.append({"text": p["text"], "severity": severity})
        elif isinstance(p, str):
            _log(f"[critic] Problem returned as plain string (old/unexpected shape), defaulting severity to 'moderate': {p[:80]}")
            normalized.append({"text": p, "severity": "moderate"})
        else:
            _log(f"[critic] Skipping unparseable problem entry: {p!r}")
    return normalized


def _normalize_verdict(critique: dict) -> dict:
    """Derives the final verdict from BOTH the score AND the severity of
    listed problems -- not the score alone. A "critical" problem always
    forces verdict="fail", regardless of what score the critic gave.
    This is the actual bug fix: previously a high score could silently
    override a critical problem the critic itself wrote down."""
    critique["problems"] = _normalize_problems(critique.get("problems"))

    has_critical = any(p["severity"] == "critical" for p in critique["problems"])
    score = critique.get("score")

    if has_critical:
        derived = "fail"
        original_verdict = critique.get("verdict")
        if original_verdict != derived:
            _log(f"[critic] Overriding self-reported verdict '{original_verdict}' (score={score}) -> 'fail' because a CRITICAL problem was listed -- score cannot override this")
        critique["verdict"] = derived
        return critique

    if score is None:
        return critique  # nothing to normalize against, leave as-is (e.g. "unavailable")

    if score >= PASS_THRESHOLD:
        derived = "pass"
    elif score < FAIL_THRESHOLD:
        derived = "fail"
    else:
        derived = "needs_improvement"

    original_verdict = critique.get("verdict")
    if original_verdict != derived:
        _log(f"[critic] Overriding self-reported verdict '{original_verdict}' -> '{derived}' based on score={score} (threshold pass>={PASS_THRESHOLD}, fail<{FAIL_THRESHOLD})")

    critique["verdict"] = derived
    return critique


from logging_config import get_logger

logger = get_logger(__name__)


def _log(msg):
    logger.info(msg)


def _log_error(msg):
    """Always recorded, even in AGENT_QUIET mode -- console verbosity is
    controlled centrally by logging_config's console handler level, not
    by skipping the call here. A Bedrock critique failure is a real
    error, not routine progress noise -- it should never be silently
    indistinguishable from a normal run just because the CLI wasn't
    started with --verbose."""
    logger.error(msg)


def _get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


CRITIC_SYSTEM_PROMPT = """You are an independent, skeptical security code reviewer.
You are reviewing a fix that a DIFFERENT AI model produced for a security
vulnerability. Your job is to catch cases where the fix LOOKS correct but
doesn't actually solve the underlying problem -- for example, adding a
`revision` parameter but setting it to "main" (which is not a real pin),
or replacing an `assert` in a way that silently changes behavior.

Be skeptical by default. A fix that merely satisfies a linter/scanner's
pattern-match without addressing the real risk should score LOW, even if
it looks superficially reasonable.

For EVERY problem you list, you must tag its severity:
- "critical": the core vulnerability is still present/exploitable, OR the
  fix introduces a new vulnerability of similar or worse severity, OR
  something explicitly security-relevant was silently left broken (e.g.
  an authentication/authorization check that still doesn't verify
  anything, a still-exploitable injection point, secrets still exposed).
  ANY critical problem means this fix CANNOT pass, no matter how good the
  rest of the change is -- so only use "critical" when you mean it.
- "moderate": a real weakness or incomplete hardening that should be
  fixed, but the original specific vulnerability being addressed is
  actually resolved (e.g. missing input validation on the new env var,
  an arbitrary but reasonable timeout value, no error handling on a
  edge case).
- "minor": style, robustness, or best-practice nitpicks that don't affect
  security correctness at all.

You will be given a list of specific finding rule IDs (e.g. "B105",
"B608") that this fix was supposed to address. For EACH rule ID you
are confident this fix actually resolves, include it in
"rule_ids_addressed". Do not include a rule ID just because it was in
the list -- only include ones you personally verified are fixed by
reading the code. If you're unsure whether a particular rule's
underlying issue is actually gone, leave it out rather than guessing --
an incomplete list here is expected and fine if you genuinely aren't
sure about one of them; a WRONG list (claiming something is fixed when
it isn't) is what actually matters, since that's what this field exists
to catch.

Respond with ONLY a JSON object, no other text, in this exact shape:
{
  "score": <integer 0-10, 10 = fully correct and safe>,
  "verdict": "<one of: 'pass', 'needs_improvement', 'fail'>",
  "problems": [
    {"text": "<specific problem>", "severity": "critical|moderate|minor"},
    ...
  ],
  "suggestions": "<concrete guidance for how to fix the problems, or empty string if score is 10>",
  "rule_ids_addressed": ["<rule ID>", ...]
}

If there are no problems at all, "problems" must be an empty list [].
If you did not verify any of the given rule IDs are fixed, "rule_ids_addressed"
must be an empty list [], not omitted.
Your own "verdict" and "score" should already be internally consistent
with the severities you assign (e.g. don't write a critical problem and
then claim "verdict": "pass" yourself) -- but note that our system will
independently enforce this regardless of what you put in "verdict".
"""


def _build_critique_prompt(relative_path: str, finding_descriptions: list, original_code: str, fixed_code: str, explanation: str, finding_rule_ids: list = None) -> str:
    findings_text = "\n".join(f"- {d}" for d in finding_descriptions)
    rule_ids = finding_rule_ids or []
    rule_ids_text = ", ".join(rule_ids) if rule_ids else "(no rule IDs provided)"
    return f"""File: {relative_path}

Rule IDs this fix must address (for the "rule_ids_addressed" field): {rule_ids_text}

Findings this fix was supposed to address:
{findings_text}

The fixing model's stated explanation of what it changed:
{explanation}

ORIGINAL (vulnerable) code:
```python
{original_code}
```

FIXED code (produced by a different AI model):
```python
{fixed_code}
```

Review whether the fix actually resolves the findings listed above, or
whether it only superficially looks like a fix. Check specifically for:
- Does it actually mitigate the risk, not just add a parameter/wrapper that looks right?
- Did it preserve existing functionality (no unrelated behavior changes)?
- Are there any new problems introduced by the fix itself?
- Are there any OTHER security-relevant functions/checks in the file
  (e.g. authentication, token verification, authorization) that were
  left broken or unaddressed, even if not explicitly listed in the
  findings above? Flag these as critical if found.

Respond with ONLY the JSON object described in your instructions."""


def _extract_bedrock_usage(response: dict) -> dict:
    """Maps Bedrock's converse() usage shape into LangFuse's expected
    usage_details shape. Bedrock nests usage under response["usage"] as
    inputTokens/outputTokens/totalTokens (dict keys) -- this is a
    different shape from Groq's response.usage.prompt_tokens/
    completion_tokens/total_tokens (object attributes), so it needs its
    own mapping rather than reusing the Groq logic. Returns None if the
    response doesn't include a usage block (defensive -- some Bedrock
    responses/models may omit it)."""
    usage = response.get("usage")
    if not usage:
        return None
    return {
        "input": usage.get("inputTokens"),
        "output": usage.get("outputTokens"),
        "total": usage.get("totalTokens"),
    }


def _call_bedrock_critic(prompt: str, relative_path: str, attempt: int = None, scan_id: str = None) -> str:
    """Makes the actual Bedrock converse() call, wrapped in a LangFuse
    'generation' observation when LangFuse is configured. Nests under
    whatever parent trace/span is currently active (e.g.
    generate_all_fixes -> fix attempt), same as groq-code-fix does in
    fix_generator.py. Raises on failure -- the caller (critique_fix)
    is responsible for catching and converting to the neutral
    'unavailable' verdict."""
    client = _get_bedrock_client()

    def _do_call():
        return client.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": CRITIC_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 2000, "temperature": 0.0},
        )

    if not LANGFUSE_ENABLED:
        response = _do_call()
        return response["output"]["message"]["content"][0]["text"]

    langfuse = get_client()
    # attempt/scan_id here match the same fields fix_generator.py's
    # groq-code-fix generation now carries -- lets you filter Langfuse
    # by scan_id and see the matching generate+critique pair for a
    # given retry attempt, not just a flat unordered list of calls.
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="bedrock-critic",
        model=BEDROCK_MODEL_ID,
        input=prompt,
        metadata={"file": relative_path, "attempt": attempt, "scan_id": scan_id},
    ) as generation:
        try:
            response = _do_call()
        except Exception as e:
            # Make the failure visible in LangFuse itself, not just the
            # console [ERROR] line -- previously a Bedrock exception here
            # was invisible in traces entirely.
            generation.update(output=f"ERROR: {e}", level="ERROR")
            raise

        raw_text = response["output"]["message"]["content"][0]["text"]
        usage_details = _extract_bedrock_usage(response)
        generation.update(output=raw_text, usage_details=usage_details)
        return raw_text


def critique_fix(relative_path: str, finding_descriptions: list, original_code: str, fixed_code: str, explanation: str, finding_rule_ids: list = None, attempt: int = None, scan_id: str = None) -> dict:
    """Returns a critique dict: {score, verdict, problems, suggestions, rule_ids_addressed}.
    problems is a list of {"text": str, "severity": "critical"|"moderate"|"minor"}.
    rule_ids_addressed is the subset of finding_rule_ids the critic says
    it actually verified are resolved -- used by fix_generator.py's
    rule-ID coverage cross-check to catch a critic that gives a "pass"
    without having actually engaged with every original finding.
    If Bedrock isn't configured or the call fails, returns a neutral
    'unavailable' result rather than blocking the pipeline -- eval is an
    enhancement, not a hard dependency for the core fix-generation flow."""
    if not CRITIC_ENABLED:
        return {"score": None, "verdict": "unavailable", "problems": [], "suggestions": "", "rule_ids_addressed": [], "error": "AWS credentials not configured"}

    try:
        prompt = _build_critique_prompt(relative_path, finding_descriptions, original_code, fixed_code, explanation, finding_rule_ids=finding_rule_ids)
        raw_text = _call_bedrock_critic(prompt, relative_path, attempt=attempt, scan_id=scan_id)

        # Reasoning models (e.g. Mistral's Magistral) wrap their chain-of-
        # thought in [THINK]...[/THINK] before the actual answer. Strip
        # that out -- we only want the final JSON verdict, not the
        # reasoning trace that precedes it.
        if "[/THINK]" in raw_text:
            raw_text = raw_text.split("[/THINK]", 1)[1]

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        parsed.setdefault("problems", [])
        parsed.setdefault("suggestions", "")
        raw_addressed = parsed.get("rule_ids_addressed")
        # Defensive: older/unexpected responses might omit this field
        # entirely (e.g. if the critic ignores the instruction), or
        # return something other than a list of strings -- treat
        # anything malformed as "reported nothing", same conservative
        # default the coverage check itself uses for a missing field.
        if isinstance(raw_addressed, list):
            parsed["rule_ids_addressed"] = [str(r) for r in raw_addressed if isinstance(r, (str, int))]
        else:
            if raw_addressed is not None:
                _log(f"[critic] rule_ids_addressed returned in unexpected shape ({type(raw_addressed).__name__}), treating as empty")
            parsed["rule_ids_addressed"] = []
        result = _normalize_verdict(parsed)

        if result.get("score") is not None:
            # Prometheus recording is unconditional (unlike the Langfuse
            # score below) -- CRITIC_SCORE feeds the Grafana "agent
            # health" panels, which should work even for a deployment
            # that never configured Langfuse at all.
            CRITIC_SCORE.observe(result["score"])

        if LANGFUSE_ENABLED and result.get("score") is not None:
            # A structured score, not just prose buried in the
            # generation's output -- this is what makes "average critic
            # score over time" / "which files needed the most retries"
            # actually queryable in Langfuse's UI instead of requiring
            # someone to read through every generation by hand.
            # Attached to the current TRACE (the whole scan's tree, set
            # up by generate_all_fixes' update_current_trace call)
            # rather than this one span, since multiple attempts/files
            # each contribute their own score to the same scan.
            get_client().score_current_trace(
                name="critic_score",
                value=result["score"],
                data_type="NUMERIC",
                comment=f"{relative_path} attempt={attempt} verdict={result.get('verdict')}",
            )

        return result

    except Exception as e:
        _log_error(f"[critic] Bedrock critique failed for {relative_path}: {e}")
        return {"score": None, "verdict": "unavailable", "problems": [], "suggestions": "", "rule_ids_addressed": [], "error": str(e)}