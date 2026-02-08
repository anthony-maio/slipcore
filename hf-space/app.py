"""
Slipstream v3: Real-Time Semantic Quantization Demo
Interactive HuggingFace Space for multi-agent AI communication protocol.
"""

import gradio as gr

from slipcore import (
    KeywordQuantizer,
    create_base_ucr,
    format_fallback,
    format_slip,
    parse_slip,
    render_human,
    __version__,
)
from slipcore.quantizer import FORCE_KEYWORDS, OBJECT_KEYWORDS, _keyword_score

# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------

UCR = create_base_ucr()
QUANTIZER = KeywordQuantizer()

FORCES = [
    "Observe", "Inform", "Ask", "Request", "Propose", "Commit",
    "Eval", "Meta", "Accept", "Reject", "Error", "Fallback",
]

OBJECTS = [
    "State", "Change", "Error", "Result", "Status", "Complete",
    "Blocked", "Progress", "Clarify", "Permission", "Resource",
    "Task", "Plan", "Review", "Help", "Cancel", "Priority",
    "Alternative", "Rollback", "Deadline", "Approve", "NeedsWork",
    "Ack", "Sync", "Handoff", "Escalate", "Abort", "Condition",
    "Defer", "Timeout", "Validation", "Generic",
]

FORCE_DESCRIPTIONS = {
    "Observe": "Passively notice state, change, or error",
    "Inform": "Report information -- status, completion, blockage, progress",
    "Ask": "Seek information -- clarification, status, permission",
    "Request": "Direct action -- task, review, help, plan",
    "Propose": "Suggest something -- plan, change, alternative",
    "Commit": "Pledge to something -- task, deadline, resource",
    "Eval": "Judge work quality -- approve, needs work",
    "Meta": "Protocol-level -- acknowledge, sync, handoff, escalate",
    "Accept": "Agree to a proposal or request",
    "Reject": "Decline a proposal or request",
    "Error": "Report system error -- timeout, resource, permission",
    "Fallback": "Content too complex for standard quantization",
}

FORCE_COLORS = {
    "Observe": "#6366f1",   # indigo
    "Inform": "#0ea5e9",    # sky
    "Ask": "#f59e0b",       # amber
    "Request": "#ef4444",   # red
    "Propose": "#8b5cf6",   # violet
    "Commit": "#22c55e",    # green
    "Eval": "#14b8a6",      # teal
    "Meta": "#64748b",      # slate
    "Accept": "#10b981",    # emerald
    "Reject": "#f43f5e",    # rose
    "Error": "#dc2626",     # red-600
    "Fallback": "#78716c",  # stone
}


# --------------------------------------------------------------------------
# Quantize tab logic
# --------------------------------------------------------------------------

def quantize_sentence(text: str, src: str, dst: str) -> str:
    """Quantize natural language into SLIP v3, showing the full TQT pipeline."""
    if not text.strip():
        return ""

    src = "".join(c for c in (src or "alice").strip() if c.isalnum())[:20] or "alice"
    dst = "".join(c for c in (dst or "bob").strip() if c.isalnum())[:20] or "bob"

    # Run quantizer
    result = QUANTIZER.quantize(text)
    force = result.force
    obj = result.obj
    confidence = result.confidence

    # Build wire
    if result.is_fallback:
        ref = QUANTIZER.fallback_store.store(text)
        wire = format_fallback(src, dst, ref)
    else:
        wire = format_slip(src, dst, force, obj)

    # Get canonical from UCR
    anchor = UCR.get_by_force_obj(force, obj)
    canonical = anchor.canonical if anchor else f"{force} {obj}"

    # Compute keyword match details for Force
    force_detail_rows = []
    for f_name, patterns in FORCE_KEYWORDS.items():
        score = _keyword_score(text, patterns)
        if score > 0:
            matched = [p for p in patterns if p.lower() in text.lower()]
            color = FORCE_COLORS.get(f_name, "#666")
            bar_width = int(score * 100)
            winner = " **<--**" if f_name == force else ""
            force_detail_rows.append(
                (score, f"| <span style='color:{color}'>`{f_name}`</span> | "
                 f"{''.join(matched[:3])} | "
                 f"<span style='display:inline-block;background:{color};width:{bar_width}px;height:12px;border-radius:3px'></span> "
                 f"{score:.2f}{winner} |")
            )
    force_detail_rows.sort(key=lambda x: x[0], reverse=True)

    # Compute keyword match details for Object
    obj_detail_rows = []
    for o_name, patterns in OBJECT_KEYWORDS.items():
        score = _keyword_score(text, patterns)
        if score > 0:
            matched = [p for p in patterns if p.lower() in text.lower()]
            bar_width = int(score * 100)
            winner = " **<--**" if o_name == obj else ""
            obj_detail_rows.append(
                (score, f"| `{o_name}` | "
                 f"{''.join(matched[:3])} | "
                 f"<span style='display:inline-block;background:#6366f1;width:{bar_width}px;height:12px;border-radius:3px'></span> "
                 f"{score:.2f}{winner} |")
            )
    obj_detail_rows.sort(key=lambda x: x[0], reverse=True)

    # Token comparison
    input_tokens = len(text.split())
    wire_tokens = len(wire.split())
    reduction = ((input_tokens - wire_tokens) / input_tokens * 100) if input_tokens > 0 else 0

    # Confidence color
    if confidence >= 0.6:
        conf_color = "#22c55e"
        conf_label = "High"
    elif confidence >= 0.3:
        conf_color = "#f59e0b"
        conf_label = "Medium"
    else:
        conf_color = "#ef4444"
        conf_label = "Low"

    force_color = FORCE_COLORS.get(force, "#666")

    # Build output
    out = []

    # Wire result (hero)
    out.append(f"### Wire Output")
    out.append(f"```")
    out.append(wire)
    out.append(f"```")
    out.append("")

    # Human-readable
    human = render_human(wire)
    out.append(f"> {human}")
    out.append("")

    # Confidence + token savings
    conf_bar_width = int(confidence * 200)
    out.append(f"| Metric | Value |")
    out.append(f"|--------|-------|")
    out.append(f"| Confidence | <span style='color:{conf_color}'>{conf_label}</span> ({confidence:.0%}) |")
    out.append(f"| Force | <span style='color:{force_color}'>{force}</span> |")
    out.append(f"| Object | {obj} |")
    out.append(f"| Canonical | \"{canonical}\" |")
    out.append(f"| Input tokens | ~{input_tokens} |")
    out.append(f"| Wire tokens | {wire_tokens} |")
    out.append(f"| Reduction | {reduction:.0f}% |")
    out.append("")

    # Stage 1: Force classification
    if force_detail_rows:
        out.append(f"<details><summary><b>Stage 1: Force Classification</b> -- matched {len(force_detail_rows)} forces</summary>")
        out.append("")
        out.append(f"| Force | Matched Keywords | Score |")
        out.append(f"|-------|-----------------|-------|")
        for _, row in force_detail_rows[:6]:
            out.append(row)
        out.append("")
        out.append(f"</details>")
        out.append("")

    # Stage 2: Object classification
    if obj_detail_rows:
        out.append(f"<details><summary><b>Stage 2: Object Classification</b> -- matched {len(obj_detail_rows)} objects</summary>")
        out.append("")
        out.append(f"| Object | Matched Keywords | Score |")
        out.append(f"|--------|-----------------|-------|")
        for _, row in obj_detail_rows[:6]:
            out.append(row)
        out.append("")
        out.append(f"</details>")
        out.append("")

    if result.is_fallback:
        out.append(f"*Fallback triggered: no keyword patterns matched with sufficient confidence. "
                    f"Raw text stored out-of-band with ref pointer.*")

    return "\n".join(out)


# --------------------------------------------------------------------------
# Encode tab logic
# --------------------------------------------------------------------------

def encode_message(src: str, dst: str, force: str, obj: str, payload: str) -> str:
    if not src or not dst or not force or not obj:
        return "Fill in source, destination, Force, and Object."

    clean_src = "".join(c for c in src.strip() if c.isalnum())[:20] or "agent"
    clean_dst = "".join(c for c in dst.strip() if c.isalnum())[:20] or "other"

    parts = ["SLIP", "v3", clean_src, clean_dst, force, obj]
    if payload.strip():
        for token in payload.strip().split():
            clean = "".join(c for c in token if c.isalnum())
            if clean:
                parts.append(clean[:30])

    wire = " ".join(parts)
    human = render_human(wire)

    input_tokens = len(parts)
    json_equiv = f'{{"from":"{src}","to":"{dst}","action":"{force}","target":"{obj}","payload":"{payload}"}}'
    json_tokens = len(json_equiv.split()) + json_equiv.count('"') + json_equiv.count(':')
    reduction = ((json_tokens - input_tokens) / json_tokens) * 100 if json_tokens > 0 else 0

    return f"""```
{wire}
```

> {human}

| Metric | SLIP v3 | JSON equivalent |
|--------|---------|-----------------|
| Tokens | **{input_tokens}** | ~{json_tokens} |
| Reduction | **{reduction:.0f}%** | -- |
"""


# --------------------------------------------------------------------------
# Decode tab logic
# --------------------------------------------------------------------------

def decode_message(wire: str) -> str:
    if not wire.strip():
        return "Enter a SLIP v3 message to decode."

    try:
        msg = parse_slip(wire.strip())
    except Exception as e:
        return f"Parse error: {e}"

    anchor = UCR.get_by_force_obj(msg.force, msg.obj)
    canonical = anchor.canonical if anchor else "unknown"
    coords = f"({', '.join(str(c) for c in anchor.coords)})" if anchor else "--"
    force_desc = FORCE_DESCRIPTIONS.get(msg.force, "Unknown")
    human = render_human(msg)

    return f"""> {human}

| Field | Value |
|-------|-------|
| Version | {msg.version} |
| Source | `{msg.src}` |
| Destination | `{msg.dst}` |
| Force | **{msg.force}** -- {force_desc} |
| Object | **{msg.obj}** |
| Canonical | \"{canonical}\" |
| Coords | `{coords}` |
| Payload | {' '.join(msg.payload) if msg.payload else '--'} |
| Fallback | {'`' + msg.fallback_ref + '`' if msg.fallback_ref else '--'} |
"""


# --------------------------------------------------------------------------
# UCR Explorer
# --------------------------------------------------------------------------

def build_ucr_table(force_filter: str) -> str:
    anchors = list(UCR.anchors.values())
    anchors.sort(key=lambda a: a.index)

    if force_filter and force_filter != "All":
        anchors = [a for a in anchors if a.force == force_filter]

    rows = []
    for a in anchors:
        color = FORCE_COLORS.get(a.force, "#666")
        coords = ", ".join(str(c) for c in a.coords)
        rows.append(
            f"| `{a.index:#06x}` | <span style='color:{color}'>{a.force}</span> | "
            f"{a.obj} | {a.canonical} | ({coords}) |"
        )

    header = f"**{len(anchors)} anchors** " + (f"(filtered: {force_filter})" if force_filter != "All" else "(all)")

    return f"""{header}

| Index | Force | Object | Canonical | Coords |
|-------|-------|--------|-----------|--------|
{chr(10).join(rows)}
"""


# --------------------------------------------------------------------------
# Example sentences for the quantizer
# --------------------------------------------------------------------------

EXAMPLES = [
    ["Please review the pull request for the auth module", "dev", "reviewer"],
    ["The database migration is complete", "worker", "manager"],
    ["I am blocked waiting for API credentials", "backend", "devops"],
    ["I suggest we switch to Redis for the session store", "architect", "team"],
    ["LGTM, approved for merge", "reviewer", "dev"],
    ["Emergency: halt all production deployments now", "sre", "allAgents"],
    ["What do you mean by 'optimize the query'?", "junior", "senior"],
    ["I will handle the database migration this sprint", "dbAdmin", "pm"],
    ["The API call to the payment provider timed out after 30s", "gateway", "monitor"],
    ["Yes, but only if we add monitoring first", "lead", "architect"],
    ["Ping, are you still there?", "coordinator", "worker"],
    ["No, that approach will not scale to our traffic levels", "cto", "engineer"],
]


# --------------------------------------------------------------------------
# Build Gradio app
# --------------------------------------------------------------------------

with gr.Blocks(
    title="Slipstream v3",
    theme=gr.themes.Soft(),
    css="""
    .hero-wire { font-size: 1.3em; font-family: monospace; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
# Slipstream v3
### Real-time semantic quantization for multi-agent AI coordination

Type a natural language sentence and watch it get quantized into a
factorized Force-Object wire message. No API keys, no GPU -- runs on
the keyword classifier built into [slipcore](https://pypi.org/project/slipcore/).

---
""")

    with gr.Tabs():

        # ---- Quantize tab (hero feature) ----
        with gr.TabItem("Quantize", id="quantize"):
            gr.Markdown("### Think -> Quantize -> Transmit")
            gr.Markdown("Enter what you want to say. The quantizer maps it to Force + Object.")

            with gr.Row():
                q_text = gr.Textbox(
                    label="Natural language input",
                    placeholder="Please review the pull request for the auth module",
                    lines=2,
                    scale=4,
                )
            with gr.Row():
                q_src = gr.Textbox(label="Source agent", value="alice", scale=1)
                q_dst = gr.Textbox(label="Destination agent", value="bob", scale=1)
                q_btn = gr.Button("Quantize", variant="primary", scale=1)

            q_output = gr.Markdown()

            q_btn.click(
                quantize_sentence,
                inputs=[q_text, q_src, q_dst],
                outputs=q_output,
            )
            q_text.submit(
                quantize_sentence,
                inputs=[q_text, q_src, q_dst],
                outputs=q_output,
            )

            gr.Markdown("#### Try these examples")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[q_text, q_src, q_dst],
                outputs=q_output,
                fn=quantize_sentence,
                cache_examples=False,
            )

        # ---- Encode tab ----
        with gr.TabItem("Encode", id="encode"):
            gr.Markdown("### Manual encoder")
            gr.Markdown("Build a wire message by picking Force and Object directly.")

            with gr.Row():
                enc_src = gr.Textbox(label="Source", value="alice", scale=1)
                enc_dst = gr.Textbox(label="Destination", value="bob", scale=1)

            with gr.Row():
                enc_force = gr.Dropdown(
                    choices=FORCES, label="Force", value="Request",
                    info="Action verb", scale=1,
                )
                enc_obj = gr.Dropdown(
                    choices=OBJECTS, label="Object", value="Review",
                    info="Domain noun", scale=1,
                )

            enc_payload = gr.Textbox(label="Payload (optional)", placeholder="auth", value="auth")
            enc_btn = gr.Button("Encode", variant="primary")
            enc_output = gr.Markdown()

            enc_btn.click(
                encode_message,
                inputs=[enc_src, enc_dst, enc_force, enc_obj, enc_payload],
                outputs=enc_output,
            )

        # ---- Decode tab ----
        with gr.TabItem("Decode", id="decode"):
            gr.Markdown("### Wire format decoder")
            gr.Markdown("Paste a SLIP v3 message to inspect it.")

            dec_input = gr.Textbox(
                label="SLIP v3 message",
                value="SLIP v3 alice bob Request Review auth",
                placeholder="SLIP v3 src dst Force Object payload...",
            )
            dec_btn = gr.Button("Decode", variant="primary")
            dec_output = gr.Markdown()

            dec_btn.click(decode_message, inputs=dec_input, outputs=dec_output)

        # ---- UCR Explorer tab ----
        with gr.TabItem("UCR Explorer", id="ucr"):
            gr.Markdown("### Universal Concept Reference")
            gr.Markdown("The 45 core anchors that form the shared semantic vocabulary.")

            ucr_filter = gr.Dropdown(
                choices=["All"] + FORCES,
                value="All",
                label="Filter by Force",
            )
            ucr_table = gr.Markdown(value=build_ucr_table("All"))

            ucr_filter.change(build_ucr_table, inputs=ucr_filter, outputs=ucr_table)

        # ---- About tab ----
        with gr.TabItem("About", id="about"):
            gr.Markdown(f"""
### What is Slipstream?

Slipstream is a protocol that performs **semantic quantization**: mapping free-form
messages onto a shared Universal Concept Reference (UCR) and transmitting factorized
intents (Force + Object) that identify structured actions.

Instead of sending `"Could you please take a look at my pull request for the authentication module changes?"` (17 tokens),
Slipstream transmits `SLIP v3 dev reviewer Request Review auth` (7 tokens).

**82% token reduction** while preserving semantic fidelity.

### How the quantizer works

The keyword quantizer runs in two stages:

1. **Force classification** -- match input against keyword patterns for each of the 12 Force tokens. Pick the highest-scoring Force.
2. **Object classification** -- match input against keyword patterns for each of the 31+ Object tokens. Pick the highest-scoring Object.

If no pattern matches above the confidence threshold, the message falls back to `Fallback Generic` with a pointer reference to the original text stored out-of-band.

For production use, swap in the embedding-based `SemanticQuantizer` from `slipcore_ml` (requires sentence-transformers).

---

### Resources

| Resource | Link |
|----------|------|
| GitHub | [github.com/anthony-maio/slipcore](https://github.com/anthony-maio/slipcore) |
| PyPI | `pip install slipcore` (v{__version__}) |
| Paper | [doi.org/10.5281/zenodo.18063451](https://doi.org/10.5281/zenodo.18063451) |
| SDK Guide | [docs/sdk-guide.md](https://github.com/anthony-maio/slipcore/blob/master/docs/sdk-guide.md) |
| Model (LoRA) | [anthonym21/slipstream-glm-z1-9b](https://huggingface.co/anthonym21/slipstream-glm-z1-9b) |
| Dataset | [anthonym21/slipstream-tqt](https://huggingface.co/datasets/anthonym21/slipstream-tqt) |

### Cost savings at scale

| Deployment | Agents | Annual JSON cost | Annual SLIP cost | Savings |
|------------|--------|------------------|------------------|---------|
| Startup | 10 | $3,600 | $650 | $2,950 |
| Scale-up | 50 | $180,000 | $32,400 | $147,600 |
| Enterprise | 1,000 | $2,500,000 | $450,000 | $2,050,000 |

---

**Citation:**
```bibtex
@misc{{maio2025slipstream,
  title={{Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination}},
  author={{Maio, Anthony}},
  year={{2025}},
  doi={{10.5281/zenodo.18063451}}
}}
```

Apache 2.0 License | [Anthony Maio](https://github.com/anthony-maio)
""")

    gr.Markdown(f"<center><sub>slipcore v{__version__} | keyword quantizer | "
                f"[source](https://github.com/anthony-maio/slipcore/tree/master/hf-space)</sub></center>")


if __name__ == "__main__":
    demo.launch()
