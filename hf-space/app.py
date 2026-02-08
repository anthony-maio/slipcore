"""
Slipstream v3: Semantic Quantization for Multi-Agent AI Communication
Interactive demo and paper showcase
"""

import gradio as gr

# Paper metadata
PAPER_TITLE = "Slipstream: Semantic Quantization for Efficient Multi-Agent Coordination"
PAPER_ABSTRACT = """As multi-agent LLM systems scale, coordination bandwidth becomes a primary cost driver:
every token spent on routing, intent framing, and redundant context is paid repeatedly across agents and turns.
Current approaches waste 40-60% of compute on coordination overhead, with communication costs scaling O(n^2) as agent counts increase.

This paper introduces Slipstream, a protocol that performs semantic quantization: mapping free-form messages onto
a shared Universal Concept Reference (UCR) and transmitting factorized intents (Force + Object) that identify structured actions.
Unlike syntactic compression (which fails due to BPE tokenizer fragmentation), Slipstream transmits natural-language
tokens that tokenize efficiently across model architectures.

**v3 Innovation:** Factorized 2-token intents (Force-Object) replace ~46 flat anchors.
This reduces the classification problem from 46-way to 12-way + 30-way, making it learnable by small models.

**Results:** 82% token reduction (41.9 -> 7.4 tokens average) while maintaining semantic fidelity."""

# Force tokens
FORCES = [
    "Observe", "Inform", "Ask", "Request", "Propose", "Commit",
    "Eval", "Meta", "Accept", "Reject", "Error", "Fallback",
]

# Object tokens
OBJECTS = [
    "State", "Change", "Error", "Result", "Status", "Complete",
    "Blocked", "Progress", "Clarify", "Permission", "Resource",
    "Task", "Plan", "Review", "Help", "Cancel", "Priority",
    "Alternative", "Rollback", "Deadline", "Approve", "NeedsWork",
    "Ack", "Sync", "Handoff", "Escalate", "Abort", "Condition",
    "Defer", "Timeout", "Validation", "Generic",
]

FORCE_DESCRIPTIONS = {
    "Observe": "Passively notice state/change/error",
    "Inform": "Report information (status, completion, blockage)",
    "Ask": "Request information (clarification, status, permission)",
    "Request": "Ask for action (task, review, help, plan)",
    "Propose": "Suggest something (plan, change, alternative)",
    "Commit": "Commit to something (task, deadline, resource)",
    "Eval": "Evaluate work (approve, needs work)",
    "Meta": "Protocol-level (acknowledge, sync, handoff)",
    "Accept": "Accept a proposal or request",
    "Reject": "Decline a proposal or request",
    "Error": "Report system error (timeout, resource, permission)",
    "Fallback": "Content too specific for standard tokens",
}

def encode_message(src: str, dst: str, force: str, obj: str, payload: str) -> str:
    """Encode a message in Slipstream v3 wire format."""
    if not src or not dst or not force or not obj:
        return "Please fill in source, destination, Force, and Object."

    # Clean tokens (alphanumeric only)
    clean_src = "".join(c for c in src.strip() if c.isalnum())[:20] or "agent"
    clean_dst = "".join(c for c in dst.strip() if c.isalnum())[:20] or "other"

    parts = ["SLIP", "v3", clean_src, clean_dst, force, obj]
    if payload.strip():
        for token in payload.strip().split():
            clean = "".join(c for c in token if c.isalnum())
            if clean:
                parts.append(clean[:30])

    wire = " ".join(parts)

    # Token comparison
    json_example = f'{{"from": "{src}", "to": "{dst}", "force": "{force.lower()}", "object": "{obj.lower()}", "payload": "{payload}"}}'

    json_tokens = len(json_example.split()) + json_example.count('"') + json_example.count(':')
    slip_tokens = len(parts)

    reduction = ((json_tokens - slip_tokens) / json_tokens) * 100 if json_tokens > 0 else 0

    return f"""**Slipstream v3 Wire Format:**
```
{wire}
```

**Estimated Tokens:** {slip_tokens}

**Equivalent JSON (~{json_tokens} tokens):**
```json
{json_example}
```

**Token Reduction:** {reduction:.0f}%
"""

def decode_message(wire: str) -> str:
    """Decode a Slipstream v3 wire format message."""
    if not wire.strip():
        return "Please enter a SLIP v3 message to decode."

    parts = wire.strip().split()

    if len(parts) < 6 or parts[0] != "SLIP" or parts[1] != "v3":
        return "Invalid format. Expected: SLIP v3 <src> <dst> <Force> <Object> [payload...]"

    src = parts[2]
    dst = parts[3]
    force = parts[4]
    obj = parts[5]
    payload = parts[6:] if len(parts) > 6 else []

    force_desc = FORCE_DESCRIPTIONS.get(force, "Unknown force")
    is_fallback = force == "Fallback"

    result = f"""**Decoded Message:**

| Field | Value |
|-------|-------|
| Version | v3 |
| Source | {src} |
| Destination | {dst} |
| Force | {force} |
| Object | {obj} |
| Force Meaning | {force_desc} |
| Payload | {' '.join(payload) if payload else '(none)'} |
| Is Fallback | {'Yes' if is_fallback else 'No'} |
"""

    if is_fallback and payload:
        result += f"\n**Fallback Ref:** `{payload[0]}` (raw text stored out-of-band)"

    return result

# Build the Gradio interface
with gr.Blocks(title="Slipstream v3 Protocol", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # Slipstream v3
    ### Factorized Semantic Quantization for Multi-Agent AI Communication

    **82% fewer tokens. Factorized Force-Object intents. Built for the AAIF ecosystem.**

    ---
    """)

    with gr.Tabs():
        with gr.TabItem("About"):
            gr.Markdown(f"""
            ## Abstract

            {PAPER_ABSTRACT}

            ---

            ## Resources

            | Resource | Link |
            |----------|------|
            | Paper (Zenodo) | [doi.org/10.5281/zenodo.18063451](https://doi.org/10.5281/zenodo.18063451) |
            | GitHub | [github.com/anthony-maio/slipcore](https://github.com/anthony-maio/slipcore) |
            | PyPI | `pip install slipcore` |
            | Model (LoRA) | [anthonym21/slipstream-glm-z1-9b](https://huggingface.co/anthonym21/slipstream-glm-z1-9b) |
            | Dataset | [anthonym21/slipstream-tqt](https://huggingface.co/datasets/anthonym21/slipstream-tqt) |

            ---

            ## v3 Wire Format

            ```
            SLIP v3 <src> <dst> <Force> <Object> [payload...]
            ```

            - **Factorized intents** - Force (action verb) + Object (domain noun)
            - **No special characters** - avoids BPE fragmentation
            - **Space-separated** - clean tokenization
            - **12 Force tokens** - closed vocabulary, easily learned
            - **30+ Object tokens** - extensible domain nouns

            ---

            ## Cost Savings at Scale

            | Deployment | Agents | Annual JSON Cost | Annual SLIP Cost | Savings |
            |------------|--------|------------------|------------------|---------|
            | Startup | 10 | $3,600 | $650 | $2,950 |
            | Scale-up | 50 | $180,000 | $32,400 | **$147,600** |
            | Enterprise | 1,000 | $2,500,000 | $450,000 | **$2,050,000** |
            """)

        with gr.TabItem("Encode"):
            gr.Markdown("## Encode a v3 Message")
            gr.Markdown("Create a Slipstream v3 wire format message from Force-Object components.")

            with gr.Row():
                src_input = gr.Textbox(label="Source Agent", placeholder="alice", value="alice")
                dst_input = gr.Textbox(label="Destination Agent", placeholder="bob", value="bob")

            with gr.Row():
                force_input = gr.Dropdown(
                    choices=FORCES,
                    label="Force (Action Verb)",
                    value="Request",
                    info="What action is being performed?"
                )
                obj_input = gr.Dropdown(
                    choices=OBJECTS,
                    label="Object (Domain Noun)",
                    value="Review",
                    info="What domain concept is the target?"
                )

            payload_input = gr.Textbox(
                label="Payload (optional, space-separated)",
                placeholder="auth module",
                value="auth"
            )

            encode_btn = gr.Button("Encode Message", variant="primary")
            encode_output = gr.Markdown()

            encode_btn.click(
                encode_message,
                inputs=[src_input, dst_input, force_input, obj_input, payload_input],
                outputs=encode_output
            )

        with gr.TabItem("Decode"):
            gr.Markdown("## Decode a v3 Message")
            gr.Markdown("Parse a Slipstream v3 wire format message.")

            wire_input = gr.Textbox(
                label="SLIP v3 Wire Message",
                placeholder="SLIP v3 alice bob Request Review auth",
                value="SLIP v3 alice bob Request Review auth"
            )

            decode_btn = gr.Button("Decode Message", variant="primary")
            decode_output = gr.Markdown()

            decode_btn.click(
                decode_message,
                inputs=wire_input,
                outputs=decode_output
            )

        with gr.TabItem("Force Tokens"):
            gr.Markdown("## Force Tokens (12 Closed Vocabulary)")
            gr.Markdown("The action verbs in the factorized intent model:")

            force_table = "\n".join([f"| `{k}` | {v} |" for k, v in FORCE_DESCRIPTIONS.items()])
            gr.Markdown(f"""
            | Force | Description |
            |-------|-------------|
            {force_table}
            """)

        with gr.TabItem("Object Tokens"):
            gr.Markdown("## Object Tokens (30+ Extensible)")
            gr.Markdown("The domain nouns in the factorized intent model:")

            obj_rows = [f"| `{o}` |" for o in OBJECTS]
            gr.Markdown(f"""
            | Object |
            |--------|
            {chr(10).join(obj_rows)}
            """)

    gr.Markdown("""
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

if __name__ == "__main__":
    demo.launch()
