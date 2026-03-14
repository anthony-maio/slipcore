from __future__ import annotations

from typing import Any

import gradio as gr
from app_logic import (
    analyze_wire,
    build_ucr_rows,
    get_langgraph_snippet,
    get_overview_metrics,
    get_resource_rows,
    get_training_guidance,
    load_example_wires,
)

TABLE_HEADERS = ["index", "force", "object", "canonical", "coords", "core", "state"]
SNIPPET_TOPICS = [
    "Boundary Encode/Decode",
    "Force:Object Router",
    "Fallback-Aware Flow",
]
GUIDANCE_TOPICS = [
    "When should I train?",
    "What does the dataset look like?",
    "What model artifacts exist?",
    "How should I evaluate first?",
]

CUSTOM_CSS = """
:root {
  --slip-bg: #111111;
  --slip-surface: #1a1a1a;
  --slip-line: rgba(255, 255, 255, 0.08);
  --slip-text: #f4efe6;
  --slip-muted: #c5b8a0;
  --slip-accent: #d18d3b;
  --slip-cool: #83c5be;
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(209, 141, 59, 0.14), transparent 28%),
    radial-gradient(circle at top right, rgba(131, 197, 190, 0.12), transparent 22%),
    var(--slip-bg);
  color: var(--slip-text);
}

.gradio-container {
  max-width: 1320px !important;
}

.hero-shell {
  border: 1px solid var(--slip-line);
  border-radius: 28px;
  padding: 28px;
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  box-shadow: 0 24px 80px rgba(0,0,0,0.24);
}

.hero-grid {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 20px;
}

.eyebrow {
  color: var(--slip-accent);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  margin-bottom: 14px;
}

.hero-title {
  font-size: 56px;
  line-height: 0.95;
  margin: 0 0 16px 0;
  letter-spacing: -0.05em;
}

.hero-copy {
  font-size: 17px;
  line-height: 1.65;
  color: var(--slip-muted);
}

.signal-card, .mini-card {
  border: 1px solid var(--slip-line);
  border-radius: 20px;
  padding: 18px;
  background: rgba(255,255,255,0.03);
}

.signal-label {
  color: var(--slip-cool);
  text-transform: uppercase;
  letter-spacing: 0.10em;
  font-size: 12px;
  margin-bottom: 10px;
}

.signal-code {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.65;
  color: var(--slip-text);
  white-space: pre-wrap;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
}

.stat-label {
  color: var(--slip-muted);
  font-size: 13px;
  margin-top: 6px;
}

.panel-copy {
  color: var(--slip-muted);
  line-height: 1.65;
}

@media (max-width: 960px) {
  .hero-grid,
  .stats-grid {
    grid-template-columns: 1fr;
  }

  .hero-title {
    font-size: 42px;
  }
}
"""


def _metrics_html() -> str:
    cards = []
    for item in get_overview_metrics():
        cards.append(
            f"""
            <div class="mini-card">
              <div class="stat-value">{item["value"]}</div>
              <div class="stat-label">{item["metric"]}</div>
            </div>
            """
        )
    return "<div class='stats-grid'>" + "".join(cards) + "</div>"


def _resources_markdown() -> str:
    rows = ["| Resource | Link |", "|---|---|"]
    for item in get_resource_rows():
        rows.append(f"| {item['resource']} | {item['link']} |")
    return "\n".join(rows)


def _render_ucr(force_filter: str, search: str) -> tuple[str, list[list[str]]]:
    rows = build_ucr_rows(force_filter=force_filter, search=search)
    summary = (
        f"Showing **{len(rows)}** anchors"
        if rows
        else "No anchors matched the current filter."
    )
    data = [[row[header] for header in TABLE_HEADERS] for row in rows]
    return summary, data


def _render_analysis(wire: str) -> tuple[str, str, dict[str, Any], str]:
    result = analyze_wire(wire)
    if result["status"] == "valid":
        status = "### Valid wire\nThis message passes Slipstream v3 validation."
        issues = "No issues."
    else:
        status = "### Invalid wire\nThis message violates one or more Slipstream v3 invariants."
        issues = "\n".join(f"- {issue}" for issue in result["issues"])
    return status, issues, result["fields"], result["human"]


def _load_example(example_type: str, selected: str | None) -> str:
    if selected:
        return selected
    examples = load_example_wires(example_type)
    return examples[0] if examples else ""


def _example_choices(example_type: str) -> tuple[dict[str, Any], str]:
    examples = load_example_wires(example_type)
    value = examples[0] if examples else ""
    return gr.update(choices=examples, value=value), value


def _render_snippet(topic: str) -> tuple[str, str]:
    copy = {
        "Boundary Encode/Decode": (
            "Add Slipstream at the handoff boundary. "
            "Keep your graph state and existing node logic intact."
        ),
        "Force:Object Router": (
            "Route on `Force:Object` once the message is decoded. "
            "This is the smallest useful production pattern."
        ),
        "Fallback-Aware Flow": (
            "Let fallback handle the long tail first. "
            "Only train later if the fallback rate is too high for your workload."
        ),
    }[topic]
    return copy, get_langgraph_snippet(topic)


with gr.Blocks(title="Slipstream Lab") as demo:
    gr.HTML(
        """
        <section class="hero-shell">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">Slipstream 3.1.1 · Hugging Face Technical Companion</div>
              <h1 class="hero-title">Explore the protocol, not just the pitch.</h1>
              <p class="hero-copy">
                This Space is the technical counterpart to the static website.
                It is designed for engineers evaluating Slipstream in real systems:
                inspect UCR anchors, validate wire messages against the shipped invariants,
                generate LangGraph integration snippets, and review the dataset/model path
                without running live inference.
              </p>
              <p class="hero-copy">
                It is CPU-first and ZeroGPU-compatible by design.
                There is no mandatory GPU path and no large-model dependency in the app itself.
              </p>
            </div>
            <div class="signal-card">
              <div class="signal-label">Wire format</div>
              <div class="signal-code">
                SLIP v3 &lt;src&gt; &lt;dst&gt; &lt;Force&gt; &lt;Object&gt; [payload...]
              </div>
              <div class="signal-label" style="margin-top: 16px;">Example</div>
              <div class="signal-code">SLIP v3 planner reviewer Request Review auth</div>
              <div class="signal-label" style="margin-top: 16px;">Why it matters</div>
              <div class="panel-copy">
                Slipstream compresses routine coordination traffic into short, explicit
                messages that are easier to route, validate, and reason about than
                repeated JSON envelopes.
              </div>
            </div>
          </div>
        </section>
        """
    )

    gr.HTML(_metrics_html())

    with gr.Tabs():
        with gr.TabItem("Overview"):
            gr.Markdown(
                """
                ## What this Space is for

                Use this Space when you want to inspect the released protocol,
                validate concrete wire examples, or understand how Slipstream
                fits into a LangGraph-style orchestration stack.

                It does **not** try to be a second homepage and it does **not**
                depend on live model inference.
                """
            )
            gr.Markdown(_resources_markdown())

        with gr.TabItem("UCR Explorer"):
            gr.Markdown(
                """
                ## Universal Concept Reference

                Browse the 45 core anchors that define the released
                Slipstream 3.1.1 semantic surface. Filter by Force or search
                across object names, canonical text, and coordinates.
                """
            )
            with gr.Row():
                force_filter = gr.Dropdown(
                    choices=["All"] + sorted({row["force"] for row in build_ucr_rows()}),
                    value="All",
                    label="Force filter",
                )
                search_box = gr.Textbox(
                    label="Search",
                    placeholder="review, timeout, handoff, 3 4 0 4",
                )
            ucr_summary = gr.Markdown()
            ucr_table = gr.Dataframe(
                headers=TABLE_HEADERS,
                datatype=["str"] * len(TABLE_HEADERS),
                interactive=False,
                wrap=True,
            )
            for event in (force_filter.change, search_box.submit):
                event(
                    _render_ucr,
                    inputs=[force_filter, search_box],
                    outputs=[ucr_summary, ucr_table],
                )
            demo.load(
                _render_ucr,
                inputs=[force_filter, search_box],
                outputs=[ucr_summary, ucr_table],
            )

        with gr.TabItem("Conformance Lab"):
            gr.Markdown(
                """
                ## Conformance Lab

                Paste a `SLIP v3` wire message or load one of the shipped
                conformance vectors. The validator below uses the library
                implementation directly, so the results match the released
                runtime behavior.
                """
            )
            with gr.Row():
                example_type = gr.Radio(
                    choices=["Valid", "Invalid"],
                    value="Valid",
                    label="Example set",
                )
                example_wire = gr.Dropdown(
                    choices=load_example_wires("Valid"),
                    value=load_example_wires("Valid")[0],
                    label="Conformance example",
                )
            wire_input = gr.Textbox(
                label="Wire message",
                lines=4,
                value=load_example_wires("Valid")[0],
            )
            analyze_btn = gr.Button("Validate and parse", variant="primary")
            status_md = gr.Markdown()
            issues_md = gr.Markdown(label="Issues")
            fields_json = gr.JSON(label="Parsed fields")
            human_md = gr.Markdown(label="Human render")

            example_type.change(
                _example_choices,
                inputs=example_type,
                outputs=[example_wire, wire_input],
            )
            example_wire.change(
                _load_example,
                inputs=[example_type, example_wire],
                outputs=wire_input,
            )
            analyze_btn.click(
                _render_analysis,
                inputs=wire_input,
                outputs=[status_md, issues_md, fields_json, human_md],
            )
            demo.load(
                _render_analysis,
                inputs=wire_input,
                outputs=[status_md, issues_md, fields_json, human_md],
            )

        with gr.TabItem("LangGraph Starter"):
            gr.Markdown(
                """
                ## LangGraph Starter

                These snippets are meant to be copied into your graph layer.
                Start with boundary encode/decode and route on `Force:Object`.
                Training is optional and should come only after you have
                measured real traffic.
                """
            )
            snippet_topic = gr.Dropdown(
                choices=SNIPPET_TOPICS,
                value=SNIPPET_TOPICS[0],
                label="Pattern",
            )
            snippet_copy = gr.Markdown()
            snippet_code = gr.Code(language="python", interactive=False)
            snippet_topic.change(
                _render_snippet,
                inputs=snippet_topic,
                outputs=[snippet_copy, snippet_code],
            )
            demo.load(
                _render_snippet,
                inputs=snippet_topic,
                outputs=[snippet_copy, snippet_code],
            )

        with gr.TabItem("Dataset / Model"):
            gr.Markdown(
                """
                ## Dataset and model path

                Slipstream can be adopted without training. This tab is here to
                clarify when training becomes useful, what the dataset contains,
                and how to think about evaluation before you fine-tune anything.
                """
            )
            guidance_topic = gr.Dropdown(
                choices=GUIDANCE_TOPICS,
                value=GUIDANCE_TOPICS[0],
                label="Question",
            )
            guidance_md = gr.Markdown()
            guidance_topic.change(get_training_guidance, inputs=guidance_topic, outputs=guidance_md)
            demo.load(get_training_guidance, inputs=guidance_topic, outputs=guidance_md)


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(
            primary_hue="amber",
            secondary_hue="stone",
            neutral_hue="zinc",
        ),
        css=CUSTOM_CSS,
    )
