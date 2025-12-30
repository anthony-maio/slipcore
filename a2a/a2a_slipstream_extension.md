# Slipstream over A2A (SLIP-A2A) Extension — Draft v0.1

This document defines a token-friendly way to carry **Slipstream (SLIP)** wire-format messages inside the **Agent2Agent (A2A)** protocol using A2A’s built-in extension mechanism.

> Design intent: **A2A stays the interoperability contract** (discovery, tasks, streaming, artifacts).  
> **Slipstream is the content encoding** that makes the *message body* cheap for LLMs to read/write.

---

## 1. Extension identity

**Extension URI (identifier):**
```text
https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1
```
This URI is used:
- in the Agent Card’s `capabilities.extensions[]` list,
- in the request “service parameter” `A2A-Extensions`,
- inside `message.extensions[]` and `message.metadata{}` for per-message opt-in.

---

## 2. Negotiation and compatibility

### 2.1 Advertising support (Agent Card)

An A2A agent that supports SLIP-A2A MUST declare it in:

```jsonc
{
  "capabilities": {
    "extensions": [
      {
        "uri": "https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1",
        "description": "Accepts Slipstream wire text in Message.parts[].text. Uses mnemonics (not hex) for token efficiency.",
        "required": false,
        "params": {
          "slipVersion": "v1",
          "ucrVersion": "1.0.0",
          "ucrHash": "sha256:<hex>",
          "preferredInputMode": "text/slip"
        }
      }
    ]
  }
}
```

**Notes**
- `required` SHOULD remain `false` for broad interoperability.
- `ucrHash` SHOULD be a stable hash of the shared UCR anchor set (see §5).

### 2.2 Client opt-in (request-level)

Clients opt into using this extension per request by sending:

```text
A2A-Extensions: https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1
```

### 2.3 Per-message usage

When sending a Message using SLIP-A2A, the client MUST include:

```jsonc
{
  "extensions": [
    "https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1"
  ],
  "metadata": {
    "https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1": {
      "slipVersion": "v1",
      "ucrVersion": "1.0.0",
      "ucrHash": "sha256:<hex>",
      "encoding": "mnemonic",
      "confidence": 0.0
    }
  }
}
```

---

## 3. Content encoding

### 3.1 Canonical encoding (token-friendly)

SLIP-A2A encodes Slipstream as **plain text** in the first `TextPart` of the A2A message:

```text
SLIP v1 <src> <dst> <anchor> [payload...]
```

Example:

```text
SLIP v1 planner reviewer RequestReview auth_module
```

### 3.2 Why “text part” (not JSON)

- This keeps the *LLM-visible* content **space-delimited** and avoids punctuation that BPE tokenizers fragment.
- The A2A envelope remains normal JSON for transport/tooling, but the part the model reads/writes is compact.

### 3.3 Attachments

If you need to send large structured payloads (patches, files, JSON blobs), SLIP-A2A RECOMMENDS:
- put the **intent** in the Slipstream text part (anchor + minimal payload tokens), and
- put heavy data in additional A2A Parts (`FilePart` or `DataPart`).

Example structure:

```jsonc
{
  "message": {
    "role": "user",
    "parts": [
      {"text": "SLIP v1 dev reviewer RequestReview"},
      {"file": {"name": "diff.patch", "mimeType": "text/x-diff", "bytes": "<base64>"}}
    ],
    "extensions": ["https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1"],
    "metadata": {
      "https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1": {
        "slipVersion": "v1",
        "ucrVersion": "1.0.0",
        "encoding": "mnemonic"
      }
    }
  }
}
```

---

## 4. Token rules (hard requirements)

To preserve token-friendliness, SLIP-A2A imposes additional constraints on the Slipstream wire text used inside A2A messages:

1. **Anchors MUST be mnemonics**, not hex IDs (no `0x...` in-wire).
2. The wire text MUST use **spaces as the only structural delimiter**.
3. The wire text MUST NOT require punctuation markers like `|`, `{}`, `=`, `@`, `#` to be parsed.
4. Payload tokens SHOULD be “safe identifiers”: `[A-Za-z0-9._-]+`.
5. If arbitrary free-form text must be sent, use the `Fallback` anchor and carry the text in a single quoted payload token, or send a second plain `text/plain` part.

---

## 5. UCR compatibility (ucrHash)

A2A agents using Slipstream MUST share a compatible UCR. SLIP-A2A standardizes a lightweight compatibility check:

- `ucrVersion`: semantic version string (e.g., `1.0.0`).
- `ucrHash`: `sha256:` of the canonicalized anchor table:
  - sort anchors by `index`
  - for each anchor include: `index|mnemonic|canonical|coords`
  - UTF-8 encode
  - SHA-256 hash

Agents SHOULD:
- reject SLIP-A2A messages where `ucrHash` mismatches (or fall back to plain A2A text mode),
- include both `ucrVersion` and `ucrHash` in Agent Card extension params.

---

## 6. Examples

### 6.1 HTTP+JSON SendMessage request

```http
POST /v1/message:send HTTP/1.1
Host: agent.example.com
Content-Type: application/json
A2A-Version: 1.0
A2A-Extensions: https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1

{
  "message": {
    "messageId": "0d9c9a1d-7c08-4d2b-9f90-5e4b6b3fd4a1",
    "role": "user",
    "parts": [{"text": "SLIP v1 planner reviewer RequestReview auth_module"}],
    "extensions": ["https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1"],
    "metadata": {
      "https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1": {
        "slipVersion": "v1",
        "ucrVersion": "1.0.0",
        "ucrHash": "sha256:<hex>",
        "encoding": "mnemonic"
      }
    }
  },
  "configuration": {
    "acceptedOutputModes": ["text/slip", "text/plain"]
  }
}
```

---

## 7. Reference implementation

See `a2a_slipstream.py` (provided alongside this doc).

---

## 8. Status

Draft v0.1. Intended as a practical AAIF / A2A RFC seed for discussion.
