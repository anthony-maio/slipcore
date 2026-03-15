from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from slipcore import (
    LangGraphSlipstreamAdapter,
    create_base_ucr,
    parse_slip,
    render_human,
    validate_wire,
)

ROOT = Path(__file__).resolve().parents[1]
VALID_VECTORS = ROOT / "spec" / "conformance" / "valid.jsonl"
INVALID_VECTORS = ROOT / "spec" / "conformance" / "invalid.jsonl"
SPACE_RAW_ASSET_BASE = "https://huggingface.co/spaces/anthonym21/slipcore/raw/main/assets"
SPACE_URL = "https://anthonym21-slipcore.hf.space/"
WEBSITE_URL = "https://slipstream.making-minds.ai"
FAVICON_URL = f"{SPACE_RAW_ASSET_BASE}/slipstream-mark.svg"
SOCIAL_CARD_URL = f"{SPACE_RAW_ASSET_BASE}/slipstream-social-card.svg"
FALLBACK_VALID_WIRES = [
    "SLIP v3 planner reviewer Request Review auth",
    "SLIP v3 ops dev Inform Status green",
    "SLIP v3 worker router Meta Ack",
    "SLIP v3 dev sre Fallback Generic ref7f3a1b2c",
]
FALLBACK_INVALID_WIRES = [
    "BAD v3 planner reviewer Request Review auth",
    "SLIP v3 planner reviewer Unknown Review auth",
    "SLIP v3 dev sre Fallback Generic",
    "SLIP v3 planner reviewer Request Review auth-module",
]


@lru_cache(maxsize=1)
def _base_ucr():
    return create_base_ucr()


@lru_cache(maxsize=1)
def _adapter() -> LangGraphSlipstreamAdapter:
    return LangGraphSlipstreamAdapter()


def build_ucr_rows(force_filter: str = "All", search: str = "") -> list[dict[str, str]]:
    query = search.strip().lower()
    rows: list[dict[str, str]] = []

    for anchor in sorted(_base_ucr(), key=lambda item: item.index):
        if force_filter != "All" and anchor.force != force_filter:
            continue

        haystack = " ".join(
            [
                anchor.force,
                anchor.obj,
                anchor.canonical,
                " ".join(str(value) for value in anchor.coords),
            ]
        ).lower()
        if query and query not in haystack:
            continue

        rows.append(
            {
                "index": f"0x{anchor.index:04x}",
                "force": anchor.force,
                "object": anchor.obj,
                "canonical": anchor.canonical,
                "coords": str(anchor.coords),
                "core": "yes" if anchor.is_core else "no",
                "state": anchor.state.value,
            }
        )

    return rows


def analyze_wire(wire: str) -> dict[str, Any]:
    wire = wire.strip()
    issues = validate_wire(wire)
    if issues:
        return {
            "status": "invalid",
            "issues": issues,
            "human": "",
            "fields": {},
        }

    message = parse_slip(wire)
    fields = {
        "version": message.version,
        "src": message.src,
        "dst": message.dst,
        "force": message.force,
        "object": message.obj,
        "payload": " ".join(message.payload),
        "fallback_ref": message.fallback_ref or "",
        "token_count": str(message.token_count_estimate),
    }

    return {
        "status": "valid",
        "issues": [],
        "human": render_human(message),
        "fields": fields,
    }


def load_example_wires(kind: str) -> list[str]:
    path = VALID_VECTORS if kind == "Valid" else INVALID_VECTORS
    fallback = FALLBACK_VALID_WIRES if kind == "Valid" else FALLBACK_INVALID_WIRES
    examples: list[str] = []
    if not path.exists():
        return fallback
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(record["wire"])
            if len(examples) == 8:
                break
    return examples or fallback


LANGGRAPH_SNIPPETS = {
    "Boundary Encode/Decode": """from typing import TypedDict
from langgraph.graph import StateGraph

from slipcore import (
    LangGraphSlipstreamAdapter,
    make_decode_node,
    make_encode_node,
)


class AgentState(TypedDict, total=False):
    thought: str
    src: str
    dst: str
    slip_wire: str
    slip_force: str
    slip_obj: str


adapter = LangGraphSlipstreamAdapter()
encode_node = make_encode_node(adapter)
decode_node = make_decode_node(adapter)

builder = StateGraph(AgentState)
builder.add_node("slip_encode", encode_node)
builder.add_node("slip_decode", decode_node)
""",
    "Force:Object Router": """from slipcore import (
    LangGraphSlipstreamAdapter,
    make_force_object_router,
)

adapter = LangGraphSlipstreamAdapter()
route = make_force_object_router(adapter)

builder.add_conditional_edges(
    "slip_decode",
    route,
    {
        "Request:Review": "review_agent",
        "Inform:Status": "status_agent",
        "Fallback:Generic": "fallback_agent",
    },
)
""",
    "Fallback-Aware Flow": """from slipcore import LangGraphSlipstreamAdapter

adapter = LangGraphSlipstreamAdapter()

wire, result = adapter.encode_thought(
    "Check kubernetes pod logs for OOMKilled events",
    src="devops",
    dst="sre",
)

decoded = adapter.decode_wire(wire)
print(decoded.message.force, decoded.message.obj)
print(decoded.fallback_text)
""",
}


def get_langgraph_snippet(topic: str) -> str:
    return LANGGRAPH_SNIPPETS[topic]


TRAINING_GUIDANCE = {
    "When should I train?": (
        "Training is optional. Start with the built-in keyword quantizer, "
        "strict wire validation, and pointer-based fallback. Measure "
        "fallback rate and routing correctness first. Train only if your "
        "workload needs higher intent recall than the rules-based path provides."
    ),
    "What does the dataset look like?": (
        "The dataset is ShareGPT-style conversation data for Think -> "
        "Quantize -> Transmit. Typical records include THOUGHT, QUANTIZE, "
        "and SLIP lines so a model can learn the protocol without hiding "
        "the reasoning step."
    ),
    "What model artifacts exist?": (
        "Reference artifacts live on Hugging Face under the anthonym21 "
        "namespace: the dataset `slipstream-tqt`, the LoRA adapter "
        "`slipstream-glm-z1-9b`, and companion merged and GGUF variants."
    ),
    "How should I evaluate first?": (
        "Build a small gold eval set from your own agent traffic. Track "
        "Force accuracy, Object accuracy, fallback rate, and downstream "
        "routing correctness before considering any fine-tuning pass."
    ),
}


def get_training_guidance(topic: str) -> str:
    return TRAINING_GUIDANCE[topic]


def get_overview_metrics() -> list[dict[str, str]]:
    return [
        {"metric": "Current release", "value": "3.1.1"},
        {"metric": "Core dependencies", "value": "0"},
        {"metric": "Passing tests", "value": "594"},
        {"metric": "Average token reduction", "value": "82%"},
    ]


def get_head_html() -> str:
    description = (
        "Inspect Slipstream 3.1.1, validate SLIP v3 wires, browse UCR anchors, "
        "and copy LangGraph starter patterns without a live model dependency."
    )
    return f"""
<meta name="description" content="{description}" />
<meta name="theme-color" content="#111111" />
<meta property="og:site_name" content="Slipstream Lab" />
<meta property="og:title" content="Slipstream Lab" />
<meta property="og:description" content="{description}" />
<meta property="og:type" content="website" />
<meta property="og:url" content="{SPACE_URL}" />
<meta property="og:image" content="{SOCIAL_CARD_URL}" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="Slipstream Lab" />
<meta name="twitter:description" content="{description}" />
<meta name="twitter:image" content="{SOCIAL_CARD_URL}" />
<meta name="twitter:url" content="{SPACE_URL}" />
<link rel="icon" type="image/svg+xml" href="{FAVICON_URL}" />
"""


def get_overview_markdown() -> str:
    return f"""
## Start here

This Space is for engineers evaluating Slipstream in a real agent stack.

1. **Inspect the protocol surface** in `UCR Explorer`.
2. **Validate concrete messages** in `Conformance Lab`.
3. **Drop the adapter into LangGraph** from `LangGraph Starter`.
4. **Use the dataset/model tab last**, after you have measured fallback rate and routing quality.

You do not need to train a model to adopt Slipstream. Start with the runtime,
route on `Force:Object`, and train only if your workload needs better
quantization than the built-in path provides.

For the release narrative, benchmarks, and positioning, use the website: {WEBSITE_URL}
"""


def get_resource_rows() -> list[dict[str, str]]:
    return [
        {"resource": "Website", "link": "https://slipstream.making-minds.ai"},
        {"resource": "GitHub", "link": "https://github.com/anthony-maio/slipcore"},
        {"resource": "PyPI", "link": "https://pypi.org/project/slipcore/"},
        {"resource": "Paper", "link": "https://doi.org/10.5281/zenodo.18063451"},
        {
            "resource": "Dataset",
            "link": "https://huggingface.co/datasets/anthonym21/slipstream-tqt",
        },
        {
            "resource": "Reference model",
            "link": "https://huggingface.co/anthonym21/slipstream-glm-z1-9b",
        },
    ]
