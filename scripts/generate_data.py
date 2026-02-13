import random
import re
import json
import uuid
import hashlib

# ==========================================
# 1. Vocabulary & Semantics (The "Ground Truth")
# ==========================================

AGENTS = ["planner", "executor", "reviewer", "db_admin", "frontend", "sre", "user_proxy"]

# Valid Semantic Maps: (Force, Object) -> Templates
# This enforces the "Factorized Intent Model" defined in the paper.
SEMANTIC_MAP = {
    ("Request", "Task"): {
        "reasoning": "Speaker is delivering a directive (Request) for a specific work item (Task).",
        "templates": [
            "Please execute the {entity} process.",
            "Run {entity} immediately.",
            "I need you to handle the {entity} ticket.",
            "Start the {entity} job."
        ]
    },
    ("Request", "Review"): {
        "reasoning": "Speaker is requesting an inspection or evaluation (Review) of a resource.",
        "templates": [
            "Please review the code for {entity}.",
            "Check {entity} for errors.",
            "Can you take a look at the {entity} PR?",
            "Audit {entity} security."
        ]
    },
    ("Request", "Help"): {
        "reasoning": "Speaker is expressing a need for assistance (Help).",
        "templates": [
            "I am stuck on {entity}, please help.",
            "Need assistance with {entity}.",
            "Can you help me fix {entity}?"
        ]
    },
    ("Inform", "Status"): {
        "reasoning": "Speaker is providing a neutral update on state (Inform Status).",
        "templates": [
            "Current status of {entity} is nominal.",
            "Update: {entity} is running.",
            "The {entity} service is healthy."
        ]
    },
    ("Inform", "Complete"): {
        "reasoning": "Speaker asserts that a task is fully finished (Inform Complete).",
        "templates": [
            "I have finished {entity}.",
            "{entity} is now complete.",
            "Done with {entity}."
        ]
    },
    ("Inform", "Blocked"): {
        "reasoning": "Speaker asserts an inability to proceed (Inform Blocked).",
        "templates": [
            "I am blocked by {entity}.",
            "Cannot proceed due to {entity} errors.",
            "Waiting on {entity} to continue."
        ]
    },
    ("Ask", "Status"): {
        "reasoning": "Speaker is querying for information about state (Ask Status).",
        "templates": [
            "What is the status of {entity}?",
            "Is {entity} finished yet?",
            "How is the {entity} migration going?"
        ]
    },
    ("Ask", "Permission"): {
        "reasoning": "Speaker is seeking authorization (Ask Permission).",
        "templates": [
            "Can I restart {entity}?",
            "Am I allowed to deploy {entity}?",
            "Do I have permission to delete {entity}?"
        ]
    },
    ("Propose", "Plan"): {
        "reasoning": "Speaker suggests a potential course of action (Propose Plan).",
        "templates": [
            "I suggest we use {entity}.",
            "Proposing a new strategy for {entity}.",
            "How about we try {entity}?"
        ]
    },
    ("Eval", "Approve"): {
        "reasoning": "Speaker gives a positive verdict (Eval Approve).",
        "templates": [
            "{entity} looks good.",
            "Approved {entity}.",
            "LGTM: {entity}."
        ]
    },
    ("Meta", "Handoff"): {
        "reasoning": "Protocol-level transfer of control (Meta Handoff).",
        "templates": [
            "Handing off {entity} to you.",
            "You take over {entity}.",
            "Transferring control of {entity}."
        ]
    }
}

# "Dirty" entities to simulate real-world input that harms tokenization
DIRTY_ENTITIES = [
    "auth_module.py", "user-login-flow", "database_schema_v2", 
    "API_gateway_timeout", "payment-service", "docker-compose.yml", 
    "k8s_pod_manifest", "error_code_500"
]

# Phrases that trigger the Fallback mechanism
FALLBACK_PHRASES = [
    "Check the kubernetes pod logs for OOMKilled events in the staging namespace.",
    "Run a grep on /var/log/syslog for 'error 500' and pipe it to a file.",
    "If the latency exceeds 500ms, retry the request with exponential backoff.",
    "The mitochondria is the powerhouse of the cell."
]

# ==========================================
# 2. Logic & Formatting Engine
# ==========================================

def clean_payload(text: str) -> str:
    """
    Enforces v3 BPE-safety (The 'Formatting' Improvement):
    1. Replaces separators (_, -, .) with spaces.
    2. Removes non-alphanumeric characters.
    3. Collapses whitespace.
    
    Input: "auth_module.py" -> Output: "auth module py"
    """
    # Replace common separators with space to preserve word boundaries
    text = re.sub(r"[_\-\.]", " ", text)
    # Remove strict non-alphanumeric chars
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Collapse whitespace and trim
    return " ".join(clean.split())

def generate_ref(text: str) -> str:
    """Generates a deterministic-looking short reference for Fallback."""
    hash_object = hashlib.md5(text.encode())
    return "ref" + hash_object.hexdigest()[:8]

def generate_entry():
    # 15% chance of Fallback (Complex/Unquantizable)
    is_fallback = random.random() < 0.15
    
    src = random.choice(AGENTS)
    dst = random.choice([a for a in AGENTS if a != src])
    
    if is_fallback:
        thought = random.choice(FALLBACK_PHRASES)
        force = "Fallback"
        obj = "Generic"
        ref = generate_ref(thought)
        
        # Formatting: Fallback payload is ALWAYS a reference token
        wire_msg = f"SLIP v3 {src} {dst} Fallback Generic {ref}"
        
        # Reasoning teaches the model to recognize complexity
        reasoning = (
            "Analysis: The input contains complex logic or syntactic instructions that do not map "
            "cleanly to the Force-Object manifold. "
            "Action: Degrade to Fallback. "
            "Formatting: Store text out-of-band and transmit pointer reference."
        )
        
    else:
        # Standard Quantization
        keys = list(SEMANTIC_MAP.keys())
        force, obj = random.choice(keys)
        data = SEMANTIC_MAP[(force, obj)]
        
        raw_entity = random.choice(DIRTY_ENTITIES)
        thought = random.choice(data["templates"]).replace("{entity}", raw_entity)
        
        # Formatting: Clean the entity
        cleaned_entity = clean_payload(raw_entity)
        
        wire_msg = f"SLIP v3 {src} {dst} {force} {obj} {cleaned_entity}"
        
        # Reasoning teaches the model strict formatting
        reasoning = (
            f"Analysis: {data['reasoning']} "
            f"Formatting: The entity '{raw_entity}' contains special characters that fragment BPE tokenizers. "
            f"It must be sanitized to '{cleaned_entity}' (alphanumeric only) for the wire."
        )

    # Output structure compatible with OpenAI/Llama fine-tuning
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a Slipstream v3 Protocol Agent. Quantize thoughts into strict Force-Object wire format."
            },
            {
                "role": "user",
                "content": f"From: {src}\nTo: {dst}\nInput: {thought}"
            },
            {
                "role": "assistant",
                "content": f"Reasoning: {reasoning}\nWire: {wire_msg}"
            }
        ]
    }

# ==========================================
# 3. Execution
# ==========================================

# Generate a batch of data
dataset = [generate_entry() for _ in range(10)]

# Print as JSONL
for entry in dataset:
    print(json.dumps(entry))