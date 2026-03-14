#!/usr/bin/env python3
"""
Minimal LangGraph + Slipstream example.

Requires:
  pip install langgraph
  pip install slipcore
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from slipcore import (
    LangGraphSlipstreamAdapter,
    make_decode_node,
    make_encode_node,
    make_force_object_router,
)


class AgentState(TypedDict, total=False):
    thought: str
    src: str
    dst: str
    slip_wire: str
    slip_force: str
    slip_obj: str
    slip_fallback_ref: str
    slip_fallback_text: str


adapter = LangGraphSlipstreamAdapter()
encode_node = make_encode_node(adapter)
decode_node = make_decode_node(adapter)
route_node = make_force_object_router(adapter)


def planner(_state: AgentState) -> AgentState:
    return {
        "thought": "Please review the authentication module",
        "src": "planner",
        "dst": "reviewer",
    }


def reviewer(state: AgentState) -> AgentState:
    return {"thought": f"Received {state['slip_force']} {state['slip_obj']}"}


def fallback_handler(state: AgentState) -> AgentState:
    return {"thought": f"Fallback text: {state.get('slip_fallback_text', '<missing>')}"}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("planner", planner)
graph_builder.add_node("slip_encode", encode_node)
graph_builder.add_node("slip_decode", decode_node)
graph_builder.add_node("reviewer", reviewer)
graph_builder.add_node("fallback_handler", fallback_handler)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "slip_encode")
graph_builder.add_edge("slip_encode", "slip_decode")
graph_builder.add_conditional_edges(
    "slip_decode",
    route_node,
    {
        "Request:Review": "reviewer",
        "Fallback:Generic": "fallback_handler",
    },
)
graph_builder.add_edge("reviewer", END)
graph_builder.add_edge("fallback_handler", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    out = graph.invoke({})
    print(out)
