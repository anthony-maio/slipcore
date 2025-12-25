#!/usr/bin/env python3
"""
Slipstream Protocol v2 - Demo

Demonstrates the key features of Slipstream:
1. Token-aligned wire format (no special characters)
2. Semantic quantization via UCR
3. Think-Quantize-Transmit pattern
4. Extension layer for local anchors
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slipcore import (
    # Protocol
    slip, decode, fallback, encode,
    # UCR
    get_default_ucr, UCRAnchor,
    # Quantizer
    quantize, think_quantize_transmit, KeywordQuantizer,
    # Extensions
    ExtensionManager,
)


def demo_wire_format():
    """Demonstrate the token-aligned wire format."""
    print("=" * 60)
    print("1. TOKEN-ALIGNED WIRE FORMAT")
    print("=" * 60)
    print()

    # Simple message
    wire = slip("alice", "bob", "RequestReview")
    print(f"Simple message:")
    print(f"  Wire: {wire}")
    print(f"  Tokens (approx): {len(wire.split())} words")
    print()

    # Message with payload
    wire = slip("planner", "executor", "RequestTask", ["auth", "refactor"])
    print(f"With payload:")
    print(f"  Wire: {wire}")
    print()

    # Decode roundtrip
    msg = decode(wire)
    print(f"Decoded:")
    print(f"  src={msg.src}, dst={msg.dst}")
    print(f"  anchor={msg.anchor.mnemonic}")
    print(f"  canonical: {msg.anchor.canonical}")
    print(f"  payload={msg.payload}")
    print()

    # Compare to JSON
    json_equiv = '{"from": "planner", "to": "executor", "type": "request", "action": "task", "payload": ["auth", "refactor"]}'
    print(f"Comparison:")
    print(f"  JSON ({len(json_equiv)} chars): {json_equiv[:50]}...")
    print(f"  SLIP ({len(wire)} chars): {wire}")
    print(f"  Savings: ~{100 - (len(wire) / len(json_equiv) * 100):.0f}% smaller")
    print()


def demo_semantic_quantization():
    """Demonstrate the Think-Quantize-Transmit pattern."""
    print("=" * 60)
    print("2. SEMANTIC QUANTIZATION (Think-Quantize-Transmit)")
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
        result = quantize(thought)
        wire = think_quantize_transmit(thought, "agent1", "agent2")

        print(f"Thought: \"{thought[:50]}...\"" if len(thought) > 50 else f"Thought: \"{thought}\"")
        print(f"  -> Anchor: {result.anchor.mnemonic}")
        print(f"  -> Confidence: {result.confidence:.2f}")
        print(f"  -> Wire: {wire}")
        print()


def demo_fallback():
    """Demonstrate fallback for unquantizable content."""
    print("=" * 60)
    print("3. FALLBACK FOR UNQUANTIZABLE CONTENT")
    print("=" * 60)
    print()

    # Content that doesn't match any anchor well
    natural_text = "check the kubernetes pod logs for OOMKilled events"

    wire = fallback("devops", "sre", natural_text)
    print(f"Natural language: \"{natural_text}\"")
    print(f"Fallback wire: {wire}")
    print()

    msg = decode(wire)
    print(f"Decoded:")
    print(f"  anchor: {msg.anchor.mnemonic}")
    print(f"  payload: {' '.join(msg.payload)}")
    print()


def demo_ucr_structure():
    """Show the UCR semantic manifold structure."""
    print("=" * 60)
    print("4. UCR SEMANTIC MANIFOLD")
    print("=" * 60)
    print()

    ucr = get_default_ucr()

    print(f"UCR Version: {ucr.version}")
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
    samples = ["RequestTask", "EvalApprove", "InformBlocked", "MetaEscalate"]
    for mnemonic in samples:
        anchor = ucr.get_by_mnemonic(mnemonic)
        if anchor:
            print(f"  {mnemonic}:")
            print(f"    Index: 0x{anchor.index:04X}")
            print(f"    Coords: {anchor.coords}")
            print(f"    Canonical: {anchor.canonical}")
    print()


def demo_extensions():
    """Demonstrate the extension layer."""
    print("=" * 60)
    print("5. EXTENSION LAYER (Local Anchors)")
    print("=" * 60)
    print()

    manager = ExtensionManager()

    # Add custom domain-specific anchor
    anchor = manager.add_extension(
        canonical="Request Kubernetes cluster scaling",
        mnemonic="RequestK8sScale",
    )
    print(f"Added extension anchor:")
    print(f"  Mnemonic: {anchor.mnemonic}")
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


def demo_token_comparison():
    """Show token savings vs JSON."""
    print("=" * 60)
    print("6. TOKEN EFFICIENCY COMPARISON")
    print("=" * 60)
    print()

    scenarios = [
        ("Task delegation",
         slip("coord", "planner", "RequestTask", ["auth"]),
         '{"from": "coordinator", "to": "planner", "type": "request", "action": "task", "target": "auth"}'),

        ("Plan approval",
         slip("reviewer", "planner", "EvalApprove"),
         '{"from": "reviewer", "to": "planner", "type": "evaluation", "result": "approved"}'),

        ("Error report",
         slip("executor", "coord", "ErrorTimeout"),
         '{"from": "executor", "to": "coordinator", "type": "error", "code": "timeout", "message": "Operation timed out"}'),

        ("Status update",
         slip("worker", "manager", "InformProgress", ["task42"]),
         '{"from": "worker", "to": "manager", "type": "inform", "status": "in_progress", "task_id": "task42"}'),
    ]

    print(f"{'Scenario':<20} {'SLIP':<10} {'JSON':<10} {'Savings':<10}")
    print("-" * 50)

    for name, slip_wire, json_wire in scenarios:
        slip_tokens = len(slip_wire.split())
        # Rough estimate: JSON tokens ≈ chars / 4 (conservative)
        json_tokens = len(json_wire) // 4
        savings = (1 - slip_tokens / json_tokens) * 100
        print(f"{name:<20} {slip_tokens:<10} ~{json_tokens:<9} {savings:.0f}%")

    print()


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  SLIPSTREAM PROTOCOL v2 - SEMANTIC QUANTIZATION")
    print("  Token-Efficient Multi-Agent Communication")
    print("=" * 60)
    print()

    demo_wire_format()
    demo_semantic_quantization()
    demo_fallback()
    demo_ucr_structure()
    demo_extensions()
    demo_token_comparison()

    print("=" * 60)
    print("Demo complete! Slipstream is ready for use.")
    print("=" * 60)
