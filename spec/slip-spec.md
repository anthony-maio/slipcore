# SLIP Specification (v0.1)

## 1. Overview

SLIP (Streamlined Intragent Protocol) is a small, typed message language for **internal communication between agents** in an agentic system.

Goals:

- Provide a **closed set of speech acts** for coordination.
- Encode messages with **frame types** and **slot IDs**, not repeated strings.
- Support **compact wire encoding** (nSLIP) suitable for LLM prompts and logs.
- Stay orthogonal to orchestration frameworks (LangGraph, LangChain, custom).

SLIP is a logical protocol. `slipcore` is a Python implementation plus a compact textual wire form (nSLIP).

---

## 2. Message model

A `SlipMessage` has:

- `conv_id` – conversation/session id (int)
- `turn`    – turn number (int)
- `src`     – source agent id (int)
- `dst`     – destination agent id (int)
- `act`     – speech act (enum `Act`)
- `frame`   – frame type (enum `FrameType`)
- `slots`   – map from `Slot` → value (IDs, small scalars, short tags)

### Acts (`Act`)

- `OBSERVE` – report new info about world/state
- `INFORM`  – derived info / belief
- `ASK`     – request information
- `REQUEST` – request a task/operation
- `PROPOSE` – propose a plan/option
- `COMMIT`  – commit to a plan/task
- `ACCEPT`  – accept a plan/request
- `REJECT`  – reject a plan/request
- `EVAL`    – evaluate a plan/result
- `ERROR`   – report an error
- `META`    – protocol/capability update

### Frame types (`FrameType`)

- `TASK`         – task definition / status
- `PLAN`         – plan with steps and estimates
- `OBSERVATION`  – observation about the environment
- `EVALUATION`   – evaluation of something
- `CONTROL`      – meta/protocol level

### Slots (`Slot`)

Each slot has a stable integer code and a single-character prefix in nSLIP.

Core ID slots:

- `GOAL_ID`         (`g`) – goal handle
- `TASK_ID`         (`k`) – task handle
- `PARENT_TASK_ID`  (`p`) – parent task handle
- `RESULT_ID`       (`r`) – result/artifact handle

Core scalar/meta slots:

- `PRIORITY` (`q`) – small int
- `SCORE`    (`s`) – numeric score (encoded as int)
- `STATUS`   (`u`) – short status string
- `ERROR_CODE` (`e`) – error code as int
- `TAG`        (`t`) – short mnemonic tag/string

Value types:

- Integer values: non-negative ints (IDs, small scalars) encoded in base62.
- Short string values: quoted strings, NOT for large text blobs.

Large artifacts (code, docs, logs) must live in external stores and be referenced by IDs here.

---

## 3. Wire encoding: nSLIP

The nSLIP wire form is a compact string consumed/emitted by LLM agents.

Shape:

```text
@a<act>|f<frame>|c<conv>|S<src>|d<dst>|T<turn>|<slot-prefix><value>|...#
```

Where:

* `@` – start marker
* `#` – end marker
* `|` – separator
* `a,f,c,S,d,T` – header field prefixes (note: uppercase S,T to avoid conflict with slot prefixes)
* `<slot-prefix>` – one character from the slot table (e.g. `g`, `k`, `r`)

### Base62 integers

Non-negative ints are encoded in base62 using:

`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`

Example:

* `0` → `0`
* `10` → `A`
* `61` → `z`
* `62` → `10`

### Strings

Short strings are encoded as:

* `"some_text"` with minimal escaping:

  * `\` → `\\`
  * `"` → `\"`

These are for **tags** or short statuses only, not for long natural-language text.

---

## 4. Examples

### REQUEST for a task

Logical message:

* `conv_id = 3`, `turn = 1`, `src = 0 (coord)`, `dst = 1 (planner)`
* `act = REQUEST`, `frame = TASK`
* slots: `GOAL_ID = 17`, `TASK_ID = 42`, `PRIORITY = 2`, `TAG = "refactor_auth"`

nSLIP:

```text
@a3|f0|c3|S0|d1|T1|gH|kg|q2|t"refactor_auth"#
```

(Actual base62 digits depend on IDs.)

---

## 5. Intended usage

* Internal messages between agents in LangGraph / LangChain / custom frameworks.
* Agents receive `SlipMessage` + access to shared stores (goals, tasks, artifacts).
* LLM prompts are constructed from these, and LLM outputs are parsed back into `SlipMessage`s.

## 6. Future extensions (non-breaking)

* Additional slots/frames (e.g. deadlines, confidence, cost).
* Binary/byte-based encodings (protobuf/CBOR) using the same Act/Frame/Slot schema.
* Schema versioning in header.
