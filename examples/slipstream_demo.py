#!/usr/bin/env python3
"""
Slipstream Protocol v3 - Demo

Demonstrates the key features of Slipstream v3:
1. Factorized Force-Object wire format (no special characters)
2. Semantic quantization via UCR
3. Think-Quantize-Transmit pattern
4. Extension layer for local anchors
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slipcore import (
    # Wire format
    format_slip, parse_slip, format_fallback, validate_wire, SlipMessage,
    # Intent model
    ForceToken, resolve_intent, from_v2_mnemonic,
    # UCR
    UCR, UCRAnchor, create_base_ucr,
    # Render
    render_human, render_log_line,
    # Events
    FallbackStore, emit, register_sink, clear_sinks,
    # Quantizer
    KeywordQuantizer,
    # Extensions
    ExtensionManager,
)


def demo_wire_format():
    """Demonstrate the factorized Force-Object wire format."""
    print("=" * 60)
    print("1. FACTORIZED FORCE-OBJECT WIRE FORMAT")
    print("=" * 60)
    print()

    # Simple message
    wire = format_slip("alice", "bob", "Request", "Review")
    print(f"Simple message:")
    print(f"  Wire: {wire}")
    print(f"  Tokens (approx): {len(wire.split())} words")
    print()

    # Message with payload
    wire = format_slip("planner", "executor", "Request", "Task", ["auth", "refactor"])
    print(f"With payload:")
    print(f"  Wire: {wire}")
    print()

    # Parse roundtrip
    msg = parse_slip(wire)
    print(f"Parsed:")
    print(f"  src={msg.src}, dst={msg.dst}")
    print(f"  Force={msg.force}, Object={msg.obj}")
    print(f"  payload={msg.payload}")
    print(f"  is_fallback={msg.is_fallback}")
    print()

    # Human-readable rendering
    print(f"Human: {render_human(msg)}")
    print(f"Log:   {render_log_line(msg)}")
    print()

    # Compare to JSON
    json_equiv = '{"from": "planner", "to": "executor", "type": "request", "action": "task", "payload": ["auth", "refactor"]}'
    print(f"Comparison:")
    print(f"  JSON ({len(json_equiv)} chars): {json_equiv[:50]}...")
    print(f"  SLIP ({len(wire)} chars): {wire}")
    print(f"  Savings: ~{100 - (len(wire) / len(json_equiv) * 100):.0f}% smaller")
    print()


def demo_intent_model():
    """Demonstrate the factorized intent model."""
    print("=" * 60)
    print("2. FACTORIZED INTENT MODEL (Force + Object)")
    print("=" * 60)
    print()

    print("Force tokens (12 closed vocabulary):")
    for ft in ForceToken:
        print(f"  {ft.value:<10} action_coord={ft.action_coord}")
    print()

    print("Intent resolution:")
    intent = resolve_intent("Request", "Plan")
    print(f"  resolve_intent('Request', 'Plan'):")
    print(f"    mnemonic: {intent.mnemonic}")
    print(f"    coords: {intent.coords}")
    print(f"    wire_tokens: {intent.wire_tokens}")
    print()

    print("V2 migration:")
    for v2 in ["RequestTask", "EvalApprove", "InformBlocked", "MetaHandoff"]:
        v3 = from_v2_mnemonic(v2)
        print(f"  {v2:<20} -> Force={v3.force.value}, Object={v3.obj.mnemonic}")
    print()


def demo_semantic_quantization():
    """Demonstrate the Think-Quantize-Transmit pattern."""
    print("=" * 60)
    print("3. SEMANTIC QUANTIZATION (Think-Quantize-Transmit)")
    print("=" * 60)
    print()

    thoughts = [
        "Please review the authentication module for security issues",
        "I've finished implementing the feature",
        "What's the current status of the deployment?",
        "I propose we use Redis for caching instead of Memcached",
        "Yes, that looks good to me - ship it!",
        "There's an error in the payment processing code",
        "I'm blocked waiting for the API credentials",
    ]

    quantizer = KeywordQuantizer()

    for thought in thoughts:
        result = quantizer.quantize(thought)
        wire = format_slip("agent1", "agent2", result.force, result.obj)

        label = f'"{thought[:50]}..."' if len(thought) > 50 else f'"{thought}"'
        print(f"Thought: {label}")
        print(f"  -> Force={result.force}, Object={result.obj}")
        print(f"  -> Confidence: {result.confidence:.2f}")
        print(f"  -> Wire: {wire}")
        print()


def demo_fallback():
    """Demonstrate fallback for unquantizable content."""
    print("=" * 60)
    print("4. POINTER-BASED FALLBACK")
    print("=" * 60)
    print()

    store = FallbackStore()
    text = "check the kubernetes pod logs for OOMKilled events"
    ref = store.store(text)

    wire = format_fallback("devops", "sre", ref)
    print(f"Natural language: \"{text}\"")
    print(f"Stored with ref: {ref}")
    print(f"Fallback wire: {wire}")
    print()

    msg = parse_slip(wire)
    print(f"Parsed:")
    print(f"  Force={msg.force}, Object={msg.obj}")
    print(f"  fallback_ref={msg.fallback_ref}")
    print(f"  is_fallback={msg.is_fallback}")
    print()

    retrieved = store.retrieve(ref)
    print(f"Retrieved from store: \"{retrieved}\"")
    print(f"Raw text NEVER on wire - only the pointer ref")
    print()


def demo_ucr_structure():
    """Show the UCR semantic manifold structure."""
    print("=" * 60)
    print("5. UCR SEMANTIC MANIFOLD")
    print("=" * 60)
    print()

    ucr = create_base_ucr()

    print(f"UCR Authority: {ucr.authority.authority_id}")
    print(f"UCR Version: {ucr.authority.ucr_version}")
    print(f"Content Hash: {ucr.content_hash()}")
    print(f"Total anchors: {len(ucr)}")
    print(f"Core anchors: {len(ucr.core_anchors())}")
    print(f"Extension anchors: {len(ucr.extension_anchors())}")
    print()

    print("Semantic Dimensions:")
    print("  0: ACTION   (observe, inform, ask, request, propose, commit, evaluate, meta)")
    print("  1: POLARITY (negative <-- neutral --> positive)")
    print("  2: DOMAIN   (task, plan, observation, evaluation, control, resource, error, general)")
    print("  3: URGENCY  (background --> normal --> critical)")
    print()

    print("Sample anchors with coordinates:")
    samples = [("Request", "Task"), ("Eval", "Approve"), ("Inform", "Blocked"), ("Meta", "Escalate")]
    for force, obj in samples:
        anchor = ucr.get_by_force_obj(force, obj)
        if anchor:
            print(f"  {force} {obj}:")
            print(f"    Index: 0x{anchor.index:04X}")
            print(f"    Coords: {anchor.coords}")
            print(f"    Canonical: {anchor.canonical}")
    print()


def demo_extensions():
    """Demonstrate the extension layer."""
    print("=" * 60)
    print("6. EXTENSION LAYER (Local Anchors)")
    print("=" * 60)
    print()

    ucr = create_base_ucr()
    manager = ExtensionManager(ucr)

    # Add custom domain-specific anchor
    anchor = manager.add_extension(
        force="Request",
        obj="K8sScale",
        canonical="Request Kubernetes cluster scaling",
        coords=(3, 4, 5, 5),  # action=request, polarity=neutral, domain=resource, urgency=normal
    )
    print(f"Added extension anchor:")
    print(f"  Force={anchor.force}, Object={anchor.obj}")
    print(f"  Index: 0x{anchor.index:04X} (extension range)")
    print(f"  Coords: {anchor.coords}")
    print()

    # Simulate fallback tracking
    print("Simulating fallback pattern tracking...")
    fallback_patterns = [
        "check terraform state",
        "check terraform state",
        "check terraform state",
        "run database migration",
        "check terraform state",
        "deploy to staging",
        "check terraform state",
    ]

    for thought in fallback_patterns:
        manager.record_fallback(thought, "dev", "ops")

    print(f"Detected patterns: {manager.fallback_tracker.get_top_patterns(3)}")
    print(f"Suggested new anchors: {manager.suggest_extensions(min_count=3)}")
    print()


def demo_validation():
    """Demonstrate wire validation."""
    print("=" * 60)
    print("7. WIRE VALIDATION")
    print("=" * 60)
    print()

    valid_msgs = [
        "SLIP v3 alice bob Request Review auth",
        "SLIP v3 planner exec Inform Complete task42",
    ]

    invalid_msgs = [
        "SLIP v1 alice bob RequestReview",
        "SLIP v3 a.b c Request Plan",
        "SLIP v3 alice bob Yeet Plan",
        "HTTP v3 alice bob Request Plan",
    ]

    for wire in valid_msgs:
        issues = validate_wire(wire)
        print(f"  {wire}")
        print(f"    -> {'VALID' if not issues else 'INVALID: ' + ', '.join(issues)}")

    for wire in invalid_msgs:
        issues = validate_wire(wire)
        print(f"  {wire}")
        print(f"    -> {'VALID' if not issues else 'INVALID: ' + ', '.join(issues)}")
    print()


def demo_token_comparison():
    """Show token savings vs JSON."""
    print("=" * 60)
    print("8. TOKEN EFFICIENCY COMPARISON")
    print("=" * 60)
    print()

    scenarios = [
        ("Task delegation",
         format_slip("coord", "planner", "Request", "Task", ["auth"]),
         '{"from": "coordinator", "to": "planner", "type": "request", "action": "task", "target": "auth"}'),

        ("Plan approval",
         format_slip("reviewer", "planner", "Eval", "Approve"),
         '{"from": "reviewer", "to": "planner", "type": "evaluation", "result": "approved"}'),

        ("Error report",
         format_slip("executor", "coord", "Error", "Timeout"),
         '{"from": "executor", "to": "coordinator", "type": "error", "code": "timeout", "message": "Operation timed out"}'),

        ("Status update",
         format_slip("worker", "manager", "Inform", "Progress", ["task42"]),
         '{"from": "worker", "to": "manager", "type": "inform", "status": "in_progress", "task_id": "task42"}'),
    ]

    print(f"{'Scenario':<20} {'SLIP':<10} {'JSON':<10} {'Savings':<10}")
    print("-" * 50)

    for name, slip_wire, json_wire in scenarios:
        slip_tokens = len(slip_wire.split())
        json_tokens = len(json_wire) // 4
        savings = (1 - slip_tokens / json_tokens) * 100
        print(f"{name:<20} {slip_tokens:<10} ~{json_tokens:<9} {savings:.0f}%")

    print()


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  SLIPSTREAM PROTOCOL v3 - FACTORIZED SEMANTIC QUANTIZATION")
    print("  Token-Efficient Multi-Agent Communication")
    print("=" * 60)
    print()

    demo_wire_format()
    demo_intent_model()
    demo_semantic_quantization()
    demo_fallback()
    demo_ucr_structure()
    demo_extensions()
    demo_validation()
    demo_token_comparison()

    print("=" * 60)
    print("Demo complete! Slipstream v3 is ready for use.")
    print("=" * 60)
