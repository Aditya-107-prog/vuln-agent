"""
agent/graph.py
--------------
Full pipeline: router -> fetch -> scan (parallel) -> enrich ->
human_review (checkpoint 1) -> report -> fix_generate ->
pr_review (checkpoint 2) -> pr_generate -> END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from agent.state import VulnScanState
from agent.nodes import (
    router_node,
    route_decision,
    github_fetch_node,
    local_read_node,
    code_scan_node,
    dep_scan_node,
    cve_enrich_node,
    human_review_node,
    human_review_decision,
    report_node,
    fix_generate_node,
    pr_review_node,
    pr_review_decision,
    pr_generate_node,
)


def build_graph():
    builder = StateGraph(VulnScanState)

    builder.add_node("router",       router_node)
    builder.add_node("github_fetch", github_fetch_node)
    builder.add_node("local_read",   local_read_node)
    builder.add_node("code_scan",    code_scan_node)
    builder.add_node("dep_scan",     dep_scan_node)
    builder.add_node("cve_enrich",   cve_enrich_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("report",       report_node)
    builder.add_node("fix_generate", fix_generate_node)
    builder.add_node("pr_review",    pr_review_node)
    builder.add_node("pr_generate",  pr_generate_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_decision,
        {"github_fetch": "github_fetch", "local_read": "local_read", "end": END}
    )

    builder.add_edge("github_fetch", "code_scan")
    builder.add_edge("github_fetch", "dep_scan")
    builder.add_edge("local_read",   "code_scan")
    builder.add_edge("local_read",   "dep_scan")

    builder.add_edge("code_scan", "cve_enrich")
    builder.add_edge("dep_scan",  "cve_enrich")

    builder.add_edge("cve_enrich", "human_review")
    builder.add_conditional_edges(
        "human_review",
        human_review_decision,
        {"report": "report", "end": END}
    )

    builder.add_edge("report", "fix_generate")
    builder.add_edge("fix_generate", "pr_review")
    builder.add_conditional_edges(
        "pr_review",
        pr_review_decision,
        {"pr_generate": "pr_generate", "end": END}
    )

    builder.add_edge("pr_generate", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()