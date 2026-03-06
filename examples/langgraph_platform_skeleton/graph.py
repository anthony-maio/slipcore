"""Concrete graph builder using Slipstream transport in LangGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from slipcore import (
    LangGraphSlipstreamAdapter,
    make_decode_node,
    make_encode_node,
    make_force_object_router,
)

from examples.langgraph_platform_skeleton.fallback_backend import (
    BackendFallbackStore,
    InMemoryFallbackBackend,
)
from examples.langgraph_platform_skeleton.routing import DEFAULT_ROUTE, FORCE_OBJECT_ROUTES
from examples.langgraph_platform_skeleton.state import AgentState


def planner(_state: AgentState) -> AgentState:
    return {
        "thought": "Please review the authentication module",
        "src": "planner",
        "dst": "reviewer",
    }


def review_agent(state: AgentState) -> AgentState:
    return {"thought": f"Processed {state['slip_force']} {state['slip_obj']}"}


def status_agent(state: AgentState) -> AgentState:
    return {"thought": f"Status path: {state['slip_wire']}"}


def fallback_agent(state: AgentState) -> AgentState:
    return {"thought": f"Fallback text: {state.get('slip_fallback_text', '<missing>')}"}


def build_graph() -> StateGraph[AgentState]:
    backend = InMemoryFallbackBackend()
    fallback_store = BackendFallbackStore(backend)
    # Adapter expects FallbackStore shape (store/retrieve), which this provides.
    adapter = LangGraphSlipstreamAdapter(fallback_store=fallback_store)  # type: ignore[arg-type]

    encode_node = make_encode_node(adapter)
    decode_node = make_decode_node(adapter)
    route_fn = make_force_object_router(adapter)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("slip_encode", encode_node)
    graph_builder.add_node("slip_decode", decode_node)
    graph_builder.add_node("review_agent", review_agent)
    graph_builder.add_node("status_agent", status_agent)
    graph_builder.add_node("fallback_agent", fallback_agent)

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "slip_encode")
    graph_builder.add_edge("slip_encode", "slip_decode")

    routes = dict(FORCE_OBJECT_ROUTES)
    routes.setdefault(DEFAULT_ROUTE, DEFAULT_ROUTE)
    graph_builder.add_conditional_edges("slip_decode", route_fn, routes)

    graph_builder.add_edge("review_agent", END)
    graph_builder.add_edge("status_agent", END)
    graph_builder.add_edge("fallback_agent", END)

    return graph_builder


if __name__ == "__main__":
    graph = build_graph().compile()
    result = graph.invoke({})
    print(result)
