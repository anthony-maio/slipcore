"""
Slipstream over A2A (SLIP-A2A) — Reference helpers

Goal: encode Slipstream wire text inside A2A Messages using A2A extensions,
while keeping the model-visible text token-friendly.

This module is intentionally small and dependency-free (stdlib only).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------
# Extension identity
# ---------------------------

SLIP_A2A_EXTENSION_URI = (
    "https://github.com/anthony-maio/slipcore/extensions/a2a-slipstream/v1"
)

# Optional (experimental) media type identifiers
SLIP_INPUT_MODE = "text/slip"
SLIP_OUTPUT_MODE = "text/slip"

# ---------------------------
# Small utilities
# ---------------------------

def new_message_id() -> str:
    """Generate a unique messageId suitable for A2A Message.messageId."""
    return str(uuid.uuid4())


def stable_ucr_hash(ucr_dict: Dict[str, Any]) -> str:
    """
    Compute a stable sha256 hash for UCR compatibility checks.

    ucr_dict format expectation (matches slipcore.ucr.UCR.save output):
      { "version": "...", "anchors": [ {index, mnemonic, canonical, coords, ...}, ... ] }

    Hash canonicalization:
      - sort anchors by index
      - per anchor: index|mnemonic|canonical|coords (coords as comma-joined ints)
      - newline separated
    """
    anchors = list(ucr_dict.get("anchors", []))
    anchors.sort(key=lambda a: int(a.get("index", 0)))

    lines: List[str] = []
    for a in anchors:
        idx = int(a.get("index", 0))
        mnemonic = str(a.get("mnemonic", ""))
        canonical = str(a.get("canonical", ""))
        coords = a.get("coords", [])
        coords_s = ",".join(str(int(x)) for x in coords) if isinstance(coords, list) else str(coords)
        lines.append(f"{idx}|{mnemonic}|{canonical}|{coords_s}")

    payload = ("\n".join(lines)).encode("utf-8")
    return "sha256:" + sha256(payload).hexdigest()


# ---------------------------
# A2A data builders
# ---------------------------

def build_agentcard_extension(
    *,
    slip_version: str = "v1",
    ucr_version: Optional[str] = None,
    ucr_hash: Optional[str] = None,
    required: bool = False,
    description: str = "Accepts Slipstream wire text in Message.parts[].text using mnemonic anchors.",
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build an AgentExtension object for inclusion in AgentCard.capabilities.extensions[].
    """
    p = dict(params or {})
    p.setdefault("slipVersion", slip_version)
    if ucr_version is not None:
        p.setdefault("ucrVersion", ucr_version)
    if ucr_hash is not None:
        p.setdefault("ucrHash", ucr_hash)
    p.setdefault("preferredInputMode", SLIP_INPUT_MODE)

    return {
        "uri": SLIP_A2A_EXTENSION_URI,
        "description": description,
        "required": bool(required),
        "params": p,
    }


def build_slip_a2a_message(
    *,
    slip_wire: str,
    role: str = "user",
    message_id: Optional[str] = None,
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    slip_version: str = "v1",
    ucr_version: Optional[str] = None,
    ucr_hash: Optional[str] = None,
    confidence: Optional[float] = None,
    include_extension: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build an A2A Message object containing Slipstream wire text.

    Minimal required fields per spec:
      - messageId
      - role
      - parts (>= 1)
    """
    mid = message_id or new_message_id()

    msg: Dict[str, Any] = {
        "messageId": mid,
        "role": role,
        "parts": [{"text": slip_wire}],
    }

    if context_id is not None:
        msg["contextId"] = context_id
    if task_id is not None:
        msg["taskId"] = task_id

    if include_extension:
        msg["extensions"] = [SLIP_A2A_EXTENSION_URI]
        meta: Dict[str, Any] = {
            "slipVersion": slip_version,
            "encoding": "mnemonic",
        }
        if ucr_version is not None:
            meta["ucrVersion"] = ucr_version
        if ucr_hash is not None:
            meta["ucrHash"] = ucr_hash
        if confidence is not None:
            meta["confidence"] = float(confidence)

        if extra_metadata:
            # merge but don't overwrite canonical fields unless explicitly given
            for k, v in extra_metadata.items():
                if k not in meta:
                    meta[k] = v

        msg["metadata"] = {SLIP_A2A_EXTENSION_URI: meta}

    return msg


def build_send_message_http_json(
    *,
    slip_wire: str,
    accepted_output_modes: Optional[Sequence[str]] = None,
    blocking: Optional[bool] = None,
    history_length: Optional[int] = None,
    a2a_version: str = "1.0",
    include_extension_header: bool = True,
    # message metadata
    role: str = "user",
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    slip_version: str = "v1",
    ucr_version: Optional[str] = None,
    ucr_hash: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build (body, headers) for HTTP+JSON binding: POST /v1/message:send

    Body is a SendMessageRequest:
      { "message": <Message>, "configuration": <SendMessageConfiguration> }
    Headers include A2A-Version and (optionally) A2A-Extensions.
    """
    msg = build_slip_a2a_message(
        slip_wire=slip_wire,
        role=role,
        context_id=context_id,
        task_id=task_id,
        slip_version=slip_version,
        ucr_version=ucr_version,
        ucr_hash=ucr_hash,
        confidence=confidence,
        include_extension=True,
    )

    cfg: Dict[str, Any] = {}
    if accepted_output_modes is not None:
        cfg["acceptedOutputModes"] = list(accepted_output_modes)
    if blocking is not None:
        cfg["blocking"] = bool(blocking)
    if history_length is not None:
        cfg["historyLength"] = int(history_length)

    body: Dict[str, Any] = {"message": msg}
    if cfg:
        body["configuration"] = cfg

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "A2A-Version": a2a_version,
    }
    if include_extension_header:
        headers["A2A-Extensions"] = SLIP_A2A_EXTENSION_URI

    return body, headers


def build_send_message_jsonrpc(
    *,
    slip_wire: str,
    rpc_id: Any = 1,
    accepted_output_modes: Optional[Sequence[str]] = None,
    blocking: Optional[bool] = None,
    history_length: Optional[int] = None,
    # service parameters may be conveyed via JSON-RPC request params (binding-specific);
    # this helper includes them in a top-level "serviceParameters" object for convenience.
    include_service_parameters: bool = True,
    a2a_version: str = "1.0",
    role: str = "user",
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    slip_version: str = "v1",
    ucr_version: Optional[str] = None,
    ucr_hash: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-RPC request object for method "SendMessage".

    NOTE: A2A’s JSON-RPC binding defines how service parameters are transmitted.
    This helper includes a "serviceParameters" dict inside params for convenience.
    """
    msg = build_slip_a2a_message(
        slip_wire=slip_wire,
        role=role,
        context_id=context_id,
        task_id=task_id,
        slip_version=slip_version,
        ucr_version=ucr_version,
        ucr_hash=ucr_hash,
        confidence=confidence,
        include_extension=True,
    )

    cfg: Dict[str, Any] = {}
    if accepted_output_modes is not None:
        cfg["acceptedOutputModes"] = list(accepted_output_modes)
    if blocking is not None:
        cfg["blocking"] = bool(blocking)
    if history_length is not None:
        cfg["historyLength"] = int(history_length)

    params: Dict[str, Any] = {"message": msg}
    if cfg:
        params["configuration"] = cfg

    if include_service_parameters:
        params["serviceParameters"] = {
            "A2A-Version": a2a_version,
            "A2A-Extensions": SLIP_A2A_EXTENSION_URI,
        }

    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "SendMessage",
        "params": params,
    }


# ---------------------------
# Decoding helpers
# ---------------------------

def extract_slip_wire_from_a2a_message(a2a_message: Dict[str, Any]) -> Optional[str]:
    """
    Extract Slipstream wire text from an A2A Message.

    Strategy:
      1) if extension declared, trust parts[0].text
      2) else, if parts[0].text starts with "SLIP ", treat as slip anyway
    """
    parts = a2a_message.get("parts") or []
    if not parts:
        return None

    first = parts[0]
    text = first.get("text") if isinstance(first, dict) else None
    if not isinstance(text, str):
        return None

    extensions = a2a_message.get("extensions") or []
    if SLIP_A2A_EXTENSION_URI in extensions:
        return text.strip()

    if text.lstrip().startswith("SLIP "):
        return text.strip()

    return None


# ---------------------------
# Pretty-print for debugging
# ---------------------------

def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)
