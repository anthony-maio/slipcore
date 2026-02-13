"""Force+Object Classifier Dataset Generator.

Generates training data for small models (282M+) that classify natural language
into Slipstream v3 Force+Object anchor pairs. Unlike the full TQT pipeline,
this focuses on the classification task only: input text -> "Force Object".

Two modes:
  - Basic: input -> "Force Object" (bare classification)
  - CoT:   input -> "THOUGHT: ... CLEAN: ... OUTPUT: Force Object payload" (chain-of-thought)

The CoT mode teaches:
  1. WHY a Force/Object was chosen (classification reasoning)
  2. HOW to sanitize entities for BPE-safe wire format (dirty -> clean)
  3. WHEN to fall back (unquantizable inputs -> Fallback Generic)

Optional data augmentation:
  - Adversarial: Borderline examples between confusable anchor pairs with
    explicit disambiguation reasoning (sharpens decision boundaries)
  - Decode: Reverse direction (SLIP -> NL) for bidirectional understanding

Usage:
    # Basic classifier (bare Force Object output)
    python -m slipcore.finetune_classifier -n 10000 --provider openrouter

    # CoT mode with reasoning + entity cleaning (recommended)
    python -m slipcore.finetune_classifier -n 10000 --cot --provider openrouter

    # Full mix: 75% CoT + 15% adversarial + 10% decode
    python -m slipcore.finetune_classifier -n 100000 --cot \\
        --adversarial-ratio 0.15 --decode-ratio 0.10 \\
        --provider openrouter --model arcee-ai/trinity-large-preview:free --resume

    # Resume after interruption
    python -m slipcore.finetune_classifier -n 50000 --cot --provider openrouter --resume
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .finetune_llm import (
    PROVIDERS,
    call_llm as _call_llm_raw,
    _count_existing_lines,
    _format_duration,
)


def call_llm(
    messages: list[dict],
    provider: str,
    model: str,
    api_key: str,
) -> str:
    """Wrapper around finetune_llm.call_llm that handles OpenRouter soft errors.

    OpenRouter sometimes returns HTTP 200 with a JSON error body instead of
    raising an HTTP error. This wrapper detects that pattern and raises so
    the retry logic can kick in.
    """
    result = _call_llm_raw(messages, provider, model, api_key)
    # OpenRouter soft errors: 200 OK but body is '{"error": ...}' which
    # call_openai_compatible parses and hits KeyError on 'choices'.
    # If we got here, call_llm_raw succeeded, but the content might be empty.
    if not result or not result.strip():
        raise RuntimeError("Empty response from API")
    return result


# ============ Valid Anchor Pairs ============

# The 45 valid Force+Object combinations from the core UCR.
VALID_ANCHORS: list[tuple[str, str]] = [
    ("Observe", "State"),
    ("Observe", "Change"),
    ("Observe", "Error"),
    ("Inform", "Result"),
    ("Inform", "Status"),
    ("Inform", "Complete"),
    ("Inform", "Blocked"),
    ("Inform", "Progress"),
    ("Ask", "Clarify"),
    ("Ask", "Status"),
    ("Ask", "Permission"),
    ("Ask", "Resource"),
    ("Request", "Task"),
    ("Request", "Plan"),
    ("Request", "Review"),
    ("Request", "Help"),
    ("Request", "Cancel"),
    ("Request", "Priority"),
    ("Request", "Resource"),
    ("Propose", "Plan"),
    ("Propose", "Change"),
    ("Propose", "Alternative"),
    ("Propose", "Rollback"),
    ("Commit", "Task"),
    ("Commit", "Deadline"),
    ("Commit", "Resource"),
    ("Eval", "Approve"),
    ("Eval", "Review"),
    ("Eval", "NeedsWork"),
    ("Eval", "Complete"),
    ("Meta", "Ack"),
    ("Meta", "Sync"),
    ("Meta", "Handoff"),
    ("Meta", "Escalate"),
    ("Meta", "Abort"),
    ("Accept", "Generic"),
    ("Reject", "Generic"),
    ("Accept", "Condition"),
    ("Meta", "Defer"),
    ("Error", "Generic"),
    ("Error", "Timeout"),
    ("Error", "Resource"),
    ("Error", "Permission"),
    ("Error", "Validation"),
    ("Fallback", "Generic"),
]

VALID_ANCHOR_SET = frozenset(VALID_ANCHORS)

# Group anchors by Force for balanced generation
ANCHORS_BY_FORCE: dict[str, list[str]] = {}
for f, o in VALID_ANCHORS:
    ANCHORS_BY_FORCE.setdefault(f, []).append(o)

# ============ Confusable Pairs (for adversarial generation) ============

# Each entry: (correct_anchor, distractor_anchor, disambiguation_hint)
# These are the pairs a 282M model is most likely to confuse.
CONFUSABLE_PAIRS: list[tuple[tuple[str, str], tuple[str, str], str]] = [
    (("Observe", "Error"), ("Error", "Generic"),
     "Watching another system fail vs sender's own operation failed"),
    (("Error", "Generic"), ("Observe", "Error"),
     "Sender's own failure vs observing external failure"),
    (("Ask", "Status"), ("Inform", "Status"),
     "Seeking information vs providing information"),
    (("Inform", "Status"), ("Ask", "Status"),
     "Providing status vs asking for status"),
    (("Ask", "Clarify"), ("Request", "Help"),
     "Wants explanation of something unclear vs wants hands-on assistance"),
    (("Request", "Help"), ("Ask", "Clarify"),
     "Needs someone to assist vs just wants clarification"),
    (("Request", "Task"), ("Propose", "Change"),
     "Directive to act vs suggestion for discussion"),
    (("Propose", "Change"), ("Request", "Task"),
     "Suggestion open to discussion vs direct assignment"),
    (("Accept", "Generic"), ("Meta", "Ack"),
     "Agreeing to a proposal vs confirming receipt of information"),
    (("Meta", "Ack"), ("Accept", "Generic"),
     "Confirming receipt only vs agreeing to do something"),
    (("Accept", "Generic"), ("Commit", "Task"),
     "Agreeing to a request vs proactively pledging to do work"),
    (("Commit", "Task"), ("Accept", "Generic"),
     "Volunteering to do work vs responding to a request"),
    (("Eval", "Approve"), ("Accept", "Generic"),
     "Quality judgment after reviewing work vs agreeing to a proposal"),
    (("Eval", "NeedsWork"), ("Reject", "Generic"),
     "Requesting revisions (resubmit) vs definitive refusal"),
    (("Reject", "Generic"), ("Eval", "NeedsWork"),
     "Definitive no vs asking for revisions"),
    (("Inform", "Complete"), ("Eval", "Complete"),
     "Worker announcing done vs reviewer confirming criteria met"),
    (("Eval", "Complete"), ("Inform", "Complete"),
     "Reviewer verifying completion vs worker reporting done"),
    (("Request", "Cancel"), ("Meta", "Abort"),
     "Normal priority stop vs emergency halt"),
    (("Meta", "Abort"), ("Request", "Cancel"),
     "Emergency termination vs routine cancellation"),
    (("Meta", "Defer"), ("Reject", "Generic"),
     "Postponing to revisit later vs definitive refusal"),
    (("Propose", "Alternative"), ("Propose", "Change"),
     "Replacing entire approach vs modifying existing one"),
    (("Inform", "Blocked"), ("Observe", "Error"),
     "Own work is blocked vs passively watching external failure"),
    (("Inform", "Progress"), ("Inform", "Status"),
     "Incremental update on ongoing work vs general state report"),
]


# ============ System Prompt for Classifier ============

CLASSIFIER_SYSTEM_PROMPT = """You are an AI agent that classifies natural language into Slipstream v3 semantic anchors.

Given a message, output the correct Force and Object tokens that best capture its intent.

## Force Tokens (12 closed vocabulary)
- Observe: Passively report something seen in an external system
- Inform: Actively share own information (results, status, completion, blockers, progress)
- Ask: Seek information, clarification, or permission (question, not command)
- Request: Direct someone to perform an action (task, review, help, plan)
- Propose: Suggest something for consideration (plan, change, alternative)
- Commit: Pledge to do something (task, deadline, resource)
- Eval: Make a quality judgment about work (approve, needs work, review in progress)
- Meta: Protocol coordination (acknowledge, sync, handoff, escalate, abort, defer)
- Accept: Agree to a proposal or request
- Reject: Refuse a proposal or request
- Error: Report a technical failure in own operation
- Fallback: Content cannot be classified into any anchor

## Response Format
Force Object"""


# ============ CoT System Prompt ============

COT_SYSTEM_PROMPT = """You are a Slipstream v3 Protocol Agent. You quantize natural language into Force+Object anchor pairs.

For each input, reason about the intent, clean any entities for BPE-safe wire format, and output the classification.

## Force Tokens (12 closed vocabulary)
Observe, Inform, Ask, Request, Propose, Commit, Eval, Meta, Accept, Reject, Error, Fallback

## Response Format
THOUGHT: <1-sentence reasoning about why this Force and Object>
CLEAN: <original entity> -> <alphanumeric only, spaces for separators>
OUTPUT: <Force> <Object> <cleaned payload tokens>

If no entity needs cleaning, omit the CLEAN line.
If the input is unquantizable (multi-intent, no clear speech act), use Fallback Generic."""


# ============ Decode System Prompt ============

DECODE_SYSTEM_PROMPT = """You interpret Slipstream v3 Force+Object classifications into natural language.

Given a classification (Force Object and optional payload tokens), explain what the sender is communicating in plain English. Keep interpretations concise (1-2 sentences)."""


# ============ Generation Prompts ============

COT_GENERATION_PROMPT = """You are generating Chain-of-Thought training data for a small (282M parameter) model that quantizes natural language into Slipstream v3 Force+Object anchor pairs.

The model must learn THREE skills:
1. **Classification**: Pick the right Force+Object from 45 valid pairs
2. **Entity cleaning**: Sanitize real-world entities for BPE-safe wire format
3. **Fallback discipline**: Recognize unquantizable inputs and degrade gracefully

## The 45 Valid Force+Object Pairs

| Force | Objects | When to use |
|-------|---------|-------------|
| Observe | State, Change, Error | Passive bystander reports on external system |
| Inform | Result, Status, Complete, Blocked, Progress | Active participant shares own info |
| Ask | Clarify, Status, Permission, Resource | Seeks information (question, not command) |
| Request | Task, Plan, Review, Help, Cancel, Priority, Resource | Directs action (command, not question) |
| Propose | Plan, Change, Alternative, Rollback | Suggests for discussion (not directive) |
| Commit | Task, Deadline, Resource | Pledges to do work or provide resources |
| Eval | Approve, Review, NeedsWork, Complete | Quality judgment after reviewing work |
| Meta | Ack, Sync, Handoff, Escalate, Abort, Defer | Protocol coordination |
| Accept | Generic, Condition | Agrees to proposal/request |
| Reject | Generic | Refuses proposal/request |
| Error | Generic, Timeout, Resource, Permission, Validation | Sender's own operation failed |
| Fallback | Generic | Cannot classify - multi-intent or no clear speech act |

## Key Distinctions

- Observe vs Inform: Observe = passive witness of external system. Inform = reporting own work/results.
- Ask vs Request: Ask = wants info ("what is X?"). Request = wants action ("do X").
- Propose vs Request: Propose = suggestion for discussion. Request = directive.
- Accept vs Commit: Accept = "yes, I agree." Commit = "I will do it."
- Accept vs Meta Ack: Accept = agreeing to proposal. Ack = confirming receipt only.
- Eval Approve vs Accept: Eval = quality judgment after review. Accept = agreeing to proposal.
- Eval NeedsWork vs Reject: NeedsWork = "revise and resubmit." Reject = "no, refused."
- Error vs Observe Error: Error = sender's own failure. Observe Error = watching another fail.
- Request Cancel vs Meta Abort: Cancel = normal stop. Abort = emergency halt.
- Inform Complete vs Eval Complete: Worker says done -> Inform. Reviewer confirms criteria -> Eval.

## Entity Cleaning Rules (BPE Safety)

Real-world inputs contain separators that fragment tokenizers. The model must clean them:
- Replace `_`, `-`, `.` with spaces: `auth_module.py` -> `auth module py`
- Remove non-alphanumeric chars: `error#500` -> `error500`
- Keep as separate words: `docker-compose.yml` -> `docker compose yml`

## Output Format for EACH Example

Generate a JSON object with these fields:
- "input": Natural language (include dirty entities like file names, error codes, etc.)
- "thought": 1 short sentence explaining WHY this Force+Object (mention the key distinction)
- "entity_raw": The dirty entity from the input (or null if none)
- "entity_clean": The BPE-safe cleaned version (or null if none)
- "force": The Force token
- "obj": The Object token

## Diversity Requirements

Vary across domains: software dev, ML/data science, DevOps, product management, research, design, security, business ops.
Use realistic dirty entities: file paths, error codes, service names, config keys, etc.
Mix formal/casual/terse/verbose phrasing.

~15% of examples should be Fallback Generic (multi-intent, no clear speech act, raw data listings).

Target these anchors: {target_anchors}

Generate exactly {batch_size} examples as a JSON array:
[
  {{"input": "Please review the auth_module.py changes", "thought": "Directing action (review), not asking a question", "entity_raw": "auth_module.py", "entity_clean": "auth module py", "force": "Request", "obj": "Review"}},
  {{"input": "I noticed the k8s_pod_manifest health check is failing", "thought": "Passive observation of external system error, not sender's own failure", "entity_raw": "k8s_pod_manifest", "entity_clean": "k8s pod manifest", "force": "Observe", "obj": "Error"}},
  {{"input": "So I was thinking about caching and also the CI is flaky and did you see the compliance email?", "thought": "Multi-intent message with 3 topics - cannot map to single anchor", "entity_raw": null, "entity_clean": null, "force": "Fallback", "obj": "Generic"}},
  ...
]

Every Force+Object pair MUST be from the 45 valid pairs. Generate exactly {batch_size} examples."""


# ============ Adversarial CoT Prompt ============

ADVERSARIAL_COT_GENERATION_PROMPT = """You are generating ADVERSARIAL training data for a small (282M) model that classifies natural language into Slipstream v3 Force+Object pairs.

These examples must be intentionally BORDERLINE — inputs that could plausibly be classified as either anchor in a confusable pair. The THOUGHT must explicitly disambiguate by explaining WHY the correct anchor wins over the distractor.

## Confusable Pairs to Generate

For each pair below, generate examples where the CORRECT anchor is the first one, but the input is phrased so the distractor seems tempting:

{confusable_pairs_text}

## Entity Cleaning Rules (BPE Safety)
- Replace `_`, `-`, `.` with spaces: `auth_module.py` -> `auth module py`
- Remove non-alphanumeric chars: `error#500` -> `error500`

## Output Format

Generate exactly {batch_size} examples as a JSON array. Spread evenly across the confusable pairs above.

[
  {{"input": "The monitoring dashboard shows the payment gateway returning 503 errors", "thought": "Reporter is observing an external system's error, not experiencing own failure - Observe Error not Error Generic", "entity_raw": "payment_gateway", "entity_clean": "payment gateway", "force": "Observe", "obj": "Error"}},
  {{"input": "I see your message about the refactor - I'll keep that in mind", "thought": "Acknowledging receipt of information without agreeing to act - Meta Ack not Accept Generic", "entity_raw": null, "entity_clean": null, "force": "Meta", "obj": "Ack"}},
  ...
]

The THOUGHT must mention BOTH anchors and explain the distinction. Every pair MUST be from the 45 valid pairs."""


# ============ Decode Generation Prompt ============

DECODE_GENERATION_PROMPT = """You are generating DECODE training data — teaching a small model to interpret Slipstream v3 classifications into natural language.

Given a Force+Object pair and optional payload, generate a natural language interpretation of what the sender is communicating.

## The 45 Valid Force+Object Pairs

| Force | Objects |
|-------|---------|
| Observe | State, Change, Error |
| Inform | Result, Status, Complete, Blocked, Progress |
| Ask | Clarify, Status, Permission, Resource |
| Request | Task, Plan, Review, Help, Cancel, Priority, Resource |
| Propose | Plan, Change, Alternative, Rollback |
| Commit | Task, Deadline, Resource |
| Eval | Approve, Review, NeedsWork, Complete |
| Meta | Ack, Sync, Handoff, Escalate, Abort, Defer |
| Accept | Generic, Condition |
| Reject | Generic |
| Error | Generic, Timeout, Resource, Permission, Validation |
| Fallback | Generic |

## Output Format

Generate exactly {batch_size} examples as a JSON array. Each has:
- "anchor_input": The Force Object + optional payload (what the model sees as input)
- "interpretation": 1-2 sentence natural language explanation

Target these anchors: {target_anchors}

[
  {{"anchor_input": "Request Review auth module", "interpretation": "Someone is asking for a code review of the auth module."}},
  {{"anchor_input": "Inform Blocked api gateway", "interpretation": "The sender reports they are blocked and cannot proceed due to an issue with the API gateway."}},
  {{"anchor_input": "Eval NeedsWork", "interpretation": "The reviewer has examined the work and determined it requires revisions before it can be approved."}},
  ...
]

Include realistic payload entities (use underscores/dots in entity names - they will be cleaned separately). Vary between examples with and without payloads. Generate exactly {batch_size} examples."""


# ============ Basic Generation Prompt ============

CLASSIFIER_GENERATION_PROMPT = """You are generating training data for a small (282M parameter) language model that classifies natural language into Slipstream v3 Force+Object anchor pairs.

The model's ONLY job is: given natural language input, output "Force Object" (two tokens).

## The 45 Valid Force+Object Pairs

| # | Force | Object | Meaning |
|---|-------|--------|---------|
| 1 | Observe | State | Report what is seen right now (passive monitor) |
| 2 | Observe | Change | Report a detected transition (something changed) |
| 3 | Observe | Error | Report a witnessed fault (third-party observer) |
| 4 | Inform | Result | Deliver computed output (sender produced it) |
| 5 | Inform | Status | Self-report on work state |
| 6 | Inform | Complete | Announce task finished |
| 7 | Inform | Blocked | Report an impediment |
| 8 | Inform | Progress | Share incremental update |
| 9 | Ask | Clarify | Seek disambiguation |
| 10 | Ask | Status | Query work/system state |
| 11 | Ask | Permission | Seek authorization ("may I?") |
| 12 | Ask | Resource | Query availability ("is it available?") |
| 13 | Request | Task | Assign work ("please do X") |
| 14 | Request | Plan | Ask someone to create a strategy |
| 15 | Request | Review | Solicit evaluation of work |
| 16 | Request | Help | Ask for assistance (sender is stuck) |
| 17 | Request | Cancel | Direct a cancellation |
| 18 | Request | Priority | Direct a priority/urgency change |
| 19 | Request | Resource | Direct resource allocation ("give us X") |
| 20 | Propose | Plan | Offer a strategy for consideration |
| 21 | Propose | Change | Suggest modifying something |
| 22 | Propose | Alternative | Suggest replacing current approach |
| 23 | Propose | Rollback | Suggest reverting to prior state |
| 24 | Commit | Task | Pledge to do work |
| 25 | Commit | Deadline | Pledge to a timeline |
| 26 | Commit | Resource | Pledge own resources |
| 27 | Eval | Approve | Positive verdict after review |
| 28 | Eval | Review | Evaluation in progress, no verdict yet |
| 29 | Eval | NeedsWork | Needs revision before approval |
| 30 | Eval | Complete | Confirm work meets all criteria |
| 31 | Meta | Ack | Confirm message receipt (no commitment) |
| 32 | Meta | Sync | Liveness/sync check ("are you there?") |
| 33 | Meta | Handoff | Transfer responsibility |
| 34 | Meta | Escalate | Raise to higher authority |
| 35 | Meta | Abort | Emergency termination |
| 36 | Accept | Generic | Agree to proposal/request (unconditional) |
| 37 | Reject | Generic | Refuse proposal/request |
| 38 | Accept | Condition | Agree with stipulations ("yes, but only if...") |
| 39 | Meta | Defer | Postpone a decision ("let's revisit later") |
| 40 | Error | Generic | Unclassified technical error |
| 41 | Error | Timeout | Time limit exceeded |
| 42 | Error | Resource | Resource not available (capacity problem) |
| 43 | Error | Permission | Authorization failure |
| 44 | Error | Validation | Data/input check failure |
| 45 | Fallback | Generic | Cannot classify, too complex/multi-intent |

IMPORTANT: Only generate pairs from this exact list of 45. No other combinations are valid.

## Decision Tree for Force Selection

Check from top to bottom, use first match:
1. Technical failure in sender's own operation? -> Error
2. Unclassifiable/multi-intent content? -> Fallback
3. Responding to a prior proposal/request? Affirmatively -> Accept, Negatively -> Reject
4. Protocol coordination (ack/sync/handoff/escalate/abort/defer)? -> Meta
5. Quality judgment about work? -> Eval
6. Pledging to do something or provide resources? -> Commit
7. Suggesting for discussion? -> Propose
8. Directing someone to act? -> Request
9. Seeking information/clarification/permission? -> Ask
10. Reporting own status/results/completion/progress/blockers? -> Inform
11. Reporting passive observation of external system? -> Observe

## Key Distinctions (the model must learn these)

- Observe vs Inform: Observe = passive bystander watching external system. Inform = active participant sharing own work/results.
- Ask vs Request: Ask = wants information ("what is X?"). Request = wants action ("do X").
- Ask Resource vs Request Resource: Ask = "is it available?" Request = "allocate it."
- Propose vs Request: Propose = suggestion for discussion. Request = directive for action.
- Accept vs Commit: Accept = "yes, I agree." Commit = "I will do it."
- Accept vs Meta Ack: Accept = agreeing to a proposal. Ack = confirming receipt of info.
- Eval Approve vs Accept: Eval Approve = quality judgment after review. Accept = agreeing to a proposal.
- Eval NeedsWork vs Reject: NeedsWork = "revise and resubmit." Reject = "no, refused."
- Error vs Observe Error: Error = sender's own operation failed. Observe Error = watching another system fail.
- Request Cancel vs Meta Abort: Cancel = normal priority stop. Abort = emergency halt.
- Meta Defer vs Reject: Defer = postpone, may revisit. Reject = definitive no.
- Commit Task vs Accept Generic: If proactively volunteering -> Commit Task. If responding to request -> Accept Generic.
- Inform Complete vs Eval Complete: Worker announces done -> Inform Complete. Reviewer confirms criteria met -> Eval Complete.

## Diversity Requirements

Generate examples across ALL of these domains:
- Software development (code, PRs, deployments, bugs, testing)
- Data science / ML (training, models, experiments, data pipelines)
- DevOps / Infrastructure (servers, containers, monitoring, incidents)
- Product management (features, roadmaps, stakeholders, priorities)
- Research (papers, experiments, findings, peer review)
- Design (mockups, UX, visual design, user testing)
- Security (audits, vulnerabilities, compliance, access)
- Business operations (hiring, budgets, vendor management)

Use varied phrasing - formal, casual, terse, verbose, with jargon, without jargon.
Include edge cases and borderline examples with the CORRECT classification.

## Your Task

Generate exactly {batch_size} training examples. For EACH example provide:
1. "input": A natural language sentence (what someone would say)
2. "force": The correct Force token
3. "obj": The correct Object token

Target these specific anchors for this batch: {target_anchors}

Output as a JSON array:
[
  {{"input": "Please review my PR for the auth module", "force": "Request", "obj": "Review"}},
  {{"input": "The CPU on prod-3 just spiked to 95%", "force": "Observe", "obj": "State"}},
  ...
]

Generate exactly {batch_size} examples. Every Force+Object pair MUST be from the 45 valid pairs listed above."""


# ============ Data Model ============

@dataclass
class ClassifierExample:
    """A single classification training example."""
    input_text: str
    force: str
    obj: str
    # CoT fields (optional, populated in --cot mode)
    thought: str = ""
    entity_raw: str = ""
    entity_clean: str = ""
    # Decode fields (optional, populated for decode examples)
    is_decode: bool = False
    decode_text: str = ""  # NL interpretation (output for decode examples)


# ============ Batch Generation ============

def _select_target_anchors(batch_size: int) -> str:
    """Select target anchors for a batch to ensure coverage.

    Cycles through all 45 anchors so no anchor is underrepresented.
    """
    # Pick enough anchors to fill the batch, cycling through all
    targets = []
    anchors = list(VALID_ANCHORS)
    random.shuffle(anchors)
    while len(targets) < batch_size:
        targets.extend(anchors)
        random.shuffle(anchors)
    targets = targets[:batch_size]

    # Format as readable list
    counts: dict[tuple[str, str], int] = {}
    for f, o in targets:
        counts[(f, o)] = counts.get((f, o), 0) + 1

    lines = []
    for (f, o), n in sorted(counts.items()):
        lines.append(f"- {f} {o}: {n} example(s)")
    return "\n".join(lines)


def generate_classifier_batch(
    batch_size: int,
    provider: str,
    model: str,
    api_key: str,
    max_retries: int = 3,
) -> list[ClassifierExample]:
    """Generate a batch of classifier examples using an LLM.

    Retries with exponential backoff on rate limit (429) errors.
    """
    target_anchors = _select_target_anchors(batch_size)
    prompt = CLASSIFIER_GENERATION_PROMPT.format(
        batch_size=batch_size,
        target_anchors=target_anchors,
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries + 1):
        try:
            response = call_llm(messages, provider, model, api_key)
            # Handle empty responses (upstream provider returned nothing)
            if not response or not response.strip():
                if attempt < max_retries:
                    wait = 2 ** attempt + random.random() * 2
                    print(f"  Empty response, retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue
                print("  Empty response after all retries")
                return []
            break
        except Exception as e:
            err = str(e)
            retriable = "429" in err or "502" in err or "503" in err or "choices" in err
            if retriable and attempt < max_retries:
                wait = 2 ** attempt + random.random() * 2
                print(f"  API error, retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries}): {err[:80]}")
                time.sleep(wait)
                continue
            raise

    # Parse JSON from response
    text = response.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        text = match.group()

    # Fix common JSON issues
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'[\x00-\x1f]', ' ', text)

    try:
        scenarios = json.loads(text)
    except json.JSONDecodeError as e:
        # Try to recover individual objects
        scenarios = []
        obj_matches = re.findall(r'\{[^{}]*\}', text)
        for obj_str in obj_matches:
            try:
                obj = json.loads(obj_str)
                if "input" in obj and "force" in obj:
                    scenarios.append(obj)
            except json.JSONDecodeError:
                continue
        if scenarios:
            print(f"  Partial recovery: {len(scenarios)} examples from malformed JSON")
        else:
            print(f"  JSON parse error: {e}")
            return []

    examples = []
    skipped_invalid = 0
    for scenario in scenarios:
        try:
            input_text = scenario["input"].strip()
            force = scenario["force"].strip()
            obj = scenario.get("obj", "Generic").strip()

            if not input_text:
                continue

            # Normalize casing
            for vf in ANCHORS_BY_FORCE:
                if vf.lower() == force.lower():
                    force = vf
                    break

            # Validate against the 45 valid pairs
            if (force, obj) not in VALID_ANCHOR_SET:
                skipped_invalid += 1
                continue

            examples.append(ClassifierExample(
                input_text=input_text,
                force=force,
                obj=obj,
            ))
        except (KeyError, TypeError, AttributeError):
            continue

    if skipped_invalid > 0:
        print(f"  Filtered {skipped_invalid} invalid anchor pairs")

    return examples


def generate_cot_batch(
    batch_size: int,
    provider: str,
    model: str,
    api_key: str,
    max_retries: int = 3,
) -> list[ClassifierExample]:
    """Generate a batch of CoT classifier examples using an LLM."""
    target_anchors = _select_target_anchors(batch_size)
    prompt = COT_GENERATION_PROMPT.format(
        batch_size=batch_size,
        target_anchors=target_anchors,
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries + 1):
        try:
            response = call_llm(messages, provider, model, api_key)
            if not response or not response.strip():
                if attempt < max_retries:
                    wait = 2 ** attempt + random.random() * 2
                    print(f"  Empty response, retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})", flush=True)
                    time.sleep(wait)
                    continue
                return []
            break
        except Exception as e:
            err = str(e)
            retriable = "429" in err or "502" in err or "503" in err or "choices" in err
            if retriable and attempt < max_retries:
                wait = 2 ** attempt + random.random() * 2
                print(f"  API error, retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries}): {err[:80]}", flush=True)
                time.sleep(wait)
                continue
            raise

    text = response.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        text = match.group()

    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'[\x00-\x1f]', ' ', text)

    try:
        scenarios = json.loads(text)
    except json.JSONDecodeError as e:
        scenarios = []
        obj_matches = re.findall(r'\{[^{}]*\}', text)
        for obj_str in obj_matches:
            try:
                obj = json.loads(obj_str)
                if "input" in obj and "force" in obj:
                    scenarios.append(obj)
            except json.JSONDecodeError:
                continue
        if scenarios:
            print(f"  Partial recovery: {len(scenarios)} examples from malformed JSON", flush=True)
        else:
            print(f"  JSON parse error: {e}", flush=True)
            return []

    examples = []
    skipped_invalid = 0
    for scenario in scenarios:
        try:
            input_text = scenario["input"].strip()
            force = scenario["force"].strip()
            obj = scenario.get("obj", "Generic").strip()

            if not input_text:
                continue

            for vf in ANCHORS_BY_FORCE:
                if vf.lower() == force.lower():
                    force = vf
                    break

            if (force, obj) not in VALID_ANCHOR_SET:
                skipped_invalid += 1
                continue

            thought = scenario.get("thought", "")
            entity_raw = scenario.get("entity_raw") or ""
            entity_clean = scenario.get("entity_clean") or ""

            # Auto-generate thought if missing
            if not thought:
                thought = f"Intent maps to {force} {obj}"

            examples.append(ClassifierExample(
                input_text=input_text,
                force=force,
                obj=obj,
                thought=thought,
                entity_raw=entity_raw,
                entity_clean=entity_clean,
            ))
        except (KeyError, TypeError, AttributeError):
            continue

    if skipped_invalid > 0:
        print(f"  Filtered {skipped_invalid} invalid anchor pairs", flush=True)

    return examples


def generate_adversarial_batch(
    batch_size: int,
    provider: str,
    model: str,
    api_key: str,
    max_retries: int = 3,
) -> list[ClassifierExample]:
    """Generate a batch of adversarial CoT examples targeting confusable pairs."""
    # Pick a random subset of confusable pairs for this batch
    pairs_sample = random.sample(
        CONFUSABLE_PAIRS, min(len(CONFUSABLE_PAIRS), batch_size // 2)
    )
    pairs_text = "\n".join(
        f"- {c[0]} {c[1]} vs {d[0]} {d[1]}: {hint}"
        for (c, d, hint) in pairs_sample
    )
    prompt = ADVERSARIAL_COT_GENERATION_PROMPT.format(
        batch_size=batch_size,
        confusable_pairs_text=pairs_text,
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries + 1):
        try:
            response = call_llm(messages, provider, model, api_key)
            if not response or not response.strip():
                if attempt < max_retries:
                    wait = 2 ** attempt + random.random() * 2
                    print(f"  Empty response, retrying in {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                return []
            break
        except Exception as e:
            err = str(e)
            retriable = "429" in err or "502" in err or "503" in err or "choices" in err
            if retriable and attempt < max_retries:
                wait = 2 ** attempt + random.random() * 2
                print(f"  API error, retrying in {wait:.0f}s: {err[:80]}", flush=True)
                time.sleep(wait)
                continue
            raise

    # Parse — same JSON recovery logic as CoT batch
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        text = match.group()
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'[\x00-\x1f]', ' ', text)

    try:
        scenarios = json.loads(text)
    except json.JSONDecodeError:
        scenarios = []
        for obj_str in re.findall(r'\{[^{}]*\}', text):
            try:
                obj = json.loads(obj_str)
                if "input" in obj and "force" in obj:
                    scenarios.append(obj)
            except json.JSONDecodeError:
                continue
        if scenarios:
            print(f"  Partial recovery: {len(scenarios)} adversarial examples", flush=True)
        else:
            return []

    examples = []
    skipped = 0
    for scenario in scenarios:
        try:
            input_text = scenario["input"].strip()
            force = scenario["force"].strip()
            obj = scenario.get("obj", "Generic").strip()
            if not input_text:
                continue
            for vf in ANCHORS_BY_FORCE:
                if vf.lower() == force.lower():
                    force = vf
                    break
            if (force, obj) not in VALID_ANCHOR_SET:
                skipped += 1
                continue
            thought = scenario.get("thought", f"Disambiguating: {force} {obj}")
            entity_raw = scenario.get("entity_raw") or ""
            entity_clean = scenario.get("entity_clean") or ""
            examples.append(ClassifierExample(
                input_text=input_text, force=force, obj=obj,
                thought=thought, entity_raw=entity_raw, entity_clean=entity_clean,
            ))
        except (KeyError, TypeError, AttributeError):
            continue

    if skipped > 0:
        print(f"  Filtered {skipped} invalid adversarial pairs", flush=True)
    return examples


def generate_decode_batch(
    batch_size: int,
    provider: str,
    model: str,
    api_key: str,
    max_retries: int = 3,
) -> list[ClassifierExample]:
    """Generate a batch of decode (SLIP -> NL) examples."""
    target_anchors = _select_target_anchors(batch_size)
    prompt = DECODE_GENERATION_PROMPT.format(
        batch_size=batch_size,
        target_anchors=target_anchors,
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries + 1):
        try:
            response = call_llm(messages, provider, model, api_key)
            if not response or not response.strip():
                if attempt < max_retries:
                    wait = 2 ** attempt + random.random() * 2
                    print(f"  Empty response, retrying in {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                return []
            break
        except Exception as e:
            err = str(e)
            retriable = "429" in err or "502" in err or "503" in err or "choices" in err
            if retriable and attempt < max_retries:
                wait = 2 ** attempt + random.random() * 2
                print(f"  API error, retrying in {wait:.0f}s: {err[:80]}", flush=True)
                time.sleep(wait)
                continue
            raise

    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        text = match.group()
    text = re.sub(r',\s*([}\]])', r'\1', text)
    text = re.sub(r'[\x00-\x1f]', ' ', text)

    try:
        scenarios = json.loads(text)
    except json.JSONDecodeError:
        scenarios = []
        for obj_str in re.findall(r'\{[^{}]*\}', text):
            try:
                obj = json.loads(obj_str)
                if "anchor_input" in obj and "interpretation" in obj:
                    scenarios.append(obj)
            except json.JSONDecodeError:
                continue
        if scenarios:
            print(f"  Partial recovery: {len(scenarios)} decode examples", flush=True)
        else:
            return []

    examples = []
    skipped = 0
    for scenario in scenarios:
        try:
            anchor_input = scenario["anchor_input"].strip()
            interpretation = scenario["interpretation"].strip()
            if not anchor_input or not interpretation:
                continue
            # Parse Force Object from anchor_input
            parts = anchor_input.split()
            if len(parts) < 2:
                continue
            force, obj = parts[0], parts[1]
            for vf in ANCHORS_BY_FORCE:
                if vf.lower() == force.lower():
                    force = vf
                    break
            if (force, obj) not in VALID_ANCHOR_SET:
                skipped += 1
                continue
            examples.append(ClassifierExample(
                input_text=anchor_input,
                force=force, obj=obj,
                is_decode=True, decode_text=interpretation,
            ))
        except (KeyError, TypeError, AttributeError):
            continue

    if skipped > 0:
        print(f"  Filtered {skipped} invalid decode pairs", flush=True)
    return examples


# ============ Output Formatters ============

def to_pairs(example: ClassifierExample) -> dict:
    """Simple input/output pairs format."""
    return {
        "input": example.input_text,
        "output": f"{example.force} {example.obj}",
    }


def to_sharegpt(example: ClassifierExample) -> dict:
    """ShareGPT format for Unsloth finetuning."""
    return {
        "conversations": [
            {"from": "system", "value": CLASSIFIER_SYSTEM_PROMPT},
            {"from": "human", "value": example.input_text},
            {"from": "gpt", "value": f"{example.force} {example.obj}"},
        ]
    }


def to_chat(example: ClassifierExample) -> dict:
    """OpenAI chat format."""
    return {
        "messages": [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": example.input_text},
            {"role": "assistant", "content": f"{example.force} {example.obj}"},
        ]
    }


def to_alpaca(example: ClassifierExample) -> dict:
    """Alpaca instruct format."""
    return {
        "instruction": "Classify this message into a Slipstream v3 Force+Object anchor pair.",
        "input": example.input_text,
        "output": f"{example.force} {example.obj}",
    }


# ---- CoT formatters ----

def _build_cot_response(example: ClassifierExample) -> str:
    """Build the CoT response string."""
    lines = [f"THOUGHT: {example.thought}"]
    if example.entity_raw and example.entity_clean:
        lines.append(f"CLEAN: {example.entity_raw} -> {example.entity_clean}")
    payload = f" {example.entity_clean}" if example.entity_clean else ""
    lines.append(f"OUTPUT: {example.force} {example.obj}{payload}")
    return "\n".join(lines)


def to_cot_sharegpt(example: ClassifierExample) -> dict:
    """ShareGPT format with CoT reasoning (recommended for Unsloth)."""
    return {
        "conversations": [
            {"from": "system", "value": COT_SYSTEM_PROMPT},
            {"from": "human", "value": example.input_text},
            {"from": "gpt", "value": _build_cot_response(example)},
        ]
    }


def to_cot_chat(example: ClassifierExample) -> dict:
    """OpenAI chat format with CoT reasoning."""
    return {
        "messages": [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": example.input_text},
            {"role": "assistant", "content": _build_cot_response(example)},
        ]
    }


def to_cot_pairs(example: ClassifierExample) -> dict:
    """Simple input/output pairs with CoT reasoning."""
    return {
        "input": example.input_text,
        "output": _build_cot_response(example),
    }


# ---- Decode formatters ----

def to_decode_sharegpt(example: ClassifierExample) -> dict:
    """ShareGPT format for decode (SLIP -> NL) examples."""
    return {
        "conversations": [
            {"from": "system", "value": DECODE_SYSTEM_PROMPT},
            {"from": "human", "value": example.input_text},
            {"from": "gpt", "value": example.decode_text},
        ]
    }


def to_decode_chat(example: ClassifierExample) -> dict:
    """OpenAI chat format for decode examples."""
    return {
        "messages": [
            {"role": "system", "content": DECODE_SYSTEM_PROMPT},
            {"role": "user", "content": example.input_text},
            {"role": "assistant", "content": example.decode_text},
        ]
    }


def to_decode_pairs(example: ClassifierExample) -> dict:
    """Simple input/output pairs for decode examples."""
    return {
        "input": example.input_text,
        "output": example.decode_text,
    }


# ---- Smart formatter that routes by example type ----

def _make_smart_converter(
    encode_fn,
    cot_fn,
    decode_fn,
    cot_mode: bool,
):
    """Return a converter that picks the right formatter per example."""
    def convert(example: ClassifierExample) -> dict:
        if example.is_decode:
            return decode_fn(example)
        if cot_mode:
            return cot_fn(example)
        return encode_fn(example)
    return convert


# ============ Main Generator ============

def generate_classifier_dataset(
    output_path: Path,
    num_examples: int = 10000,
    provider: str = "openrouter",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    format: str = "sharegpt",
    batch_size: int = 50,
    max_workers: int = 4,
    delay_between_batches: float = 0.5,
    resume: bool = False,
    cot: bool = False,
    adversarial_ratio: float = 0.0,
    decode_ratio: float = 0.0,
) -> int:
    """Generate a Force+Object classification dataset using LLM APIs.

    Args:
        output_path: Where to save the JSONL file.
        num_examples: Target number of examples.
        provider: API provider.
        model: Model name (uses provider default if not specified).
        api_key: API key (reads from env if not specified).
        format: Output format (pairs, sharegpt, chat, alpaca).
        batch_size: Examples per API call (higher = more efficient).
        max_workers: Parallel API calls.
        delay_between_batches: Seconds between batches.
        resume: Continue from existing file.
        cot: Enable Chain-of-Thought mode (reasoning + entity cleaning).
        adversarial_ratio: Fraction of batches that are adversarial (0.0-1.0).
        decode_ratio: Fraction of batches that are decode SLIP->NL (0.0-1.0).

    Returns:
        Total number of examples in the output file.
    """
    import os

    # Load .env file if present (no dependency needed)
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDERS.keys())}")

    if api_key is None:
        env_key = PROVIDERS[provider]["env_key"]
        api_key = os.environ.get(env_key)
        if not api_key:
            raise ValueError(f"Set {env_key} environment variable or pass --api-key")

    if model is None:
        model = PROVIDERS[provider]["default_model"]

    # Handle resume
    existing_count = 0
    file_mode = "w"
    if resume and output_path.exists():
        existing_count = _count_existing_lines(output_path)
        if existing_count >= num_examples:
            print(f"Resume: {output_path} already has {existing_count} examples "
                  f"(target: {num_examples}). Nothing to do.")
            return existing_count
        file_mode = "a"
        print(f"Resume: found {existing_count} existing examples in {output_path}")

    remaining = num_examples - existing_count
    num_batches = (remaining + batch_size - 1) // batch_size

    # Determine mix label
    mix_parts = []
    if cot:
        mix_parts.append("CoT")
    else:
        mix_parts.append("basic")
    if adversarial_ratio > 0:
        mix_parts.append(f"{adversarial_ratio:.0%} adversarial")
    if decode_ratio > 0:
        mix_parts.append(f"{decode_ratio:.0%} decode")
    mode_label = " + ".join(mix_parts)

    print(f"Generating {remaining} classifier examples ({mode_label}) using {provider}/{model}"
          + (f" (resuming from {existing_count})" if existing_count > 0 else ""), flush=True)
    print(f"Format: {format} | Batch size: {batch_size} | Workers: {max_workers}", flush=True)

    # Build the primary batch generator (encode direction)
    if cot:
        primary_generator = generate_cot_batch
    else:
        primary_generator = generate_classifier_batch

    # Build smart converter that routes encode/decode examples to correct formatter
    fmt_map = {
        "sharegpt": (to_sharegpt, to_cot_sharegpt, to_decode_sharegpt),
        "chat": (to_chat, to_cot_chat, to_decode_chat),
        "pairs": (to_pairs, to_cot_pairs, to_decode_pairs),
        "alpaca": (to_alpaca, to_cot_pairs, to_decode_pairs),
    }
    encode_fn, cot_fn, decode_fn = fmt_map.get(format, fmt_map["sharegpt"])
    converter = _make_smart_converter(encode_fn, cot_fn, decode_fn, cot)

    written_count = 0
    dist: dict[str, int] = {}
    type_counts = {"encode": 0, "adversarial": 0, "decode": 0}
    start_time = time.monotonic()
    completed_batches = 0

    # Open file for incremental writing (never lose progress)
    fh = open(output_path, file_mode, encoding="utf-8")

    def _flush_batch(batch: list[ClassifierExample], batch_type: str) -> int:
        """Write a batch to disk immediately. Returns count written."""
        nonlocal written_count
        space = remaining - written_count
        to_write = batch[:space]
        for ex in to_write:
            formatted = converter(ex)
            fh.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            key = f"{ex.force} {ex.obj}"
            suffix = " [decode]" if ex.is_decode else ""
            dist[key + suffix] = dist.get(key + suffix, 0) + 1
        written_count += len(to_write)
        type_counts[batch_type] = type_counts.get(batch_type, 0) + len(to_write)
        fh.flush()
        return len(to_write)

    def _pick_batch_type() -> str:
        """Randomly pick batch type according to configured ratios."""
        r = random.random()
        if r < adversarial_ratio:
            return "adversarial"
        if r < adversarial_ratio + decode_ratio:
            return "decode"
        return "encode"

    def _log(msg: str) -> None:
        print(msg, flush=True)

    try:
        batch_num = 0
        while written_count < remaining:
            wave_size = min(max_workers, num_batches - batch_num)
            if wave_size <= 0:
                break

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for _ in range(wave_size):
                    batch_type = _pick_batch_type()
                    if batch_type == "adversarial":
                        gen = generate_adversarial_batch
                    elif batch_type == "decode":
                        gen = generate_decode_batch
                    else:
                        gen = primary_generator
                    fut = executor.submit(
                        gen,
                        batch_size=batch_size,
                        provider=provider,
                        model=model,
                        api_key=api_key,
                    )
                    futures[fut] = batch_type

                for future in as_completed(futures):
                    batch_num += 1
                    completed_batches += 1
                    batch_type = futures[future]
                    try:
                        batch = future.result()
                        n_written = _flush_batch(batch, batch_type)
                        total = existing_count + written_count
                        type_tag = f" [{batch_type}]" if batch_type != "encode" else ""
                        _log(f"Batch {completed_batches}/{num_batches}{type_tag}: "
                             f"+{n_written} examples (total: {total})")
                    except Exception as e:
                        _log(f"Batch {completed_batches}/{num_batches} failed: {e}")

            # Progress report after each wave
            if completed_batches % 10 == 0 and written_count > 0:
                elapsed = time.monotonic() - start_time
                rate = written_count / elapsed
                eta = (remaining - written_count) / rate if rate > 0 else 0
                total = existing_count + written_count
                _log(f"  -- {total}/{num_examples} | "
                     f"{_format_duration(elapsed)} elapsed | "
                     f"{rate:.1f} ex/s | "
                     f"ETA {_format_duration(eta)}")

            time.sleep(delay_between_batches)
    finally:
        fh.close()

    # Report
    print(f"\nAnchor distribution ({len(dist)} unique anchors):")
    for anchor, count in sorted(dist.items(), key=lambda x: -x[1])[:10]:
        print(f"  {anchor}: {count}")
    if len(dist) > 10:
        print(f"  ... and {len(dist) - 10} more")
    missing = [f"{f} {o}" for f, o in VALID_ANCHORS if f"{f} {o}" not in dist]
    if missing:
        print(f"  Missing anchors ({len(missing)}): {', '.join(missing[:5])}"
              + ("..." if len(missing) > 5 else ""))

    # Type mix report
    if adversarial_ratio > 0 or decode_ratio > 0:
        print(f"\nExample mix: encode={type_counts['encode']}, "
              f"adversarial={type_counts['adversarial']}, "
              f"decode={type_counts['decode']}")

    total_count = existing_count + written_count
    elapsed_total = time.monotonic() - start_time
    print(f"\nWrote {written_count} classifier examples to {output_path} "
          f"in {_format_duration(elapsed_total)}")
    if existing_count > 0:
        print(f"Total examples in file: {total_count}")
    return total_count


# ============ CLI ============

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Force+Object classifier training data for small models"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("slipstream_classifier.jsonl"),
    )
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="openrouter",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-f", "--format",
        choices=["pairs", "sharegpt", "chat", "alpaca"],
        default="sharegpt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Examples per API call (default 50, higher = cheaper but less diverse)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing file",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Chain-of-Thought mode: include reasoning traces and entity cleaning",
    )
    parser.add_argument(
        "--adversarial-ratio",
        type=float,
        default=0.0,
        help="Fraction of batches that are adversarial disambiguation (0.0-1.0, e.g. 0.15)",
    )
    parser.add_argument(
        "--decode-ratio",
        type=float,
        default=0.0,
        help="Fraction of batches that are decode SLIP->NL (0.0-1.0, e.g. 0.10)",
    )

    args = parser.parse_args()

    try:
        count = generate_classifier_dataset(
            output_path=args.output,
            num_examples=args.num_examples,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            format=args.format,
            batch_size=args.batch_size,
            max_workers=args.workers,
            resume=args.resume,
            cot=args.cot,
            adversarial_ratio=args.adversarial_ratio,
            decode_ratio=args.decode_ratio,
        )
        print(f"\nDone! {count} total classifier examples")
        print(f"Ready for finetuning a 282M model with Unsloth or similar")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
