from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0.2
)

# ============================================================
# Schemas
# ============================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str] = Field(min_length=3, max_length=6)
    target_words: int

    tags: List[str] = []
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design"
    ] = "explainer"

    constraints: List[str] = []
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = []
    max_results_per_query: int = 5


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = []


class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = []


# ============================================================
# Graph State
# ============================================================

class State(TypedDict):

    topic: str

    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    as_of: str
    recency_days: int

    sections: Annotated[List[tuple[int, str]], operator.add]

    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# ============================================================
# Router
# ============================================================

ROUTER_SYSTEM = """
You are a routing module for a technical blog planner.

Decide if research is needed.

Modes
closed_book
hybrid
open_book
"""

def router_node(state: State):

    decider = llm.with_structured_output(RouterDecision)

    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}")
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days
    }


def route_next(state: State):
    if state["needs_research"]:
        return "research"
    return "orchestrator"


# ============================================================
# Research
# ============================================================

def research_node(state: State):
    return {"evidence": []}


# ============================================================
# Orchestrator
# ============================================================

ORCH_SYSTEM = """
You are a senior technical writer.

Create a blog plan with 5 to 9 sections.
"""

def orchestrator_node(state: State):

    planner = llm.with_structured_output(Plan)

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}")
        ]
    )

    return {"plan": plan}


# ============================================================
# Fanout
# ============================================================

def fanout(state: State):

    plan = state["plan"]

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "plan": plan.model_dump()
            }
        )
        for task in plan.tasks
    ]


# ============================================================
# Worker
# ============================================================

WORKER_SYSTEM = """
Write one markdown section for a technical blog.
"""

def worker_node(payload: dict):

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])

    bullets = "\n".join(task.bullets)

    section = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=f"""
Blog title: {plan.blog_title}

Section: {task.title}

Goal: {task.goal}

Bullets:
{bullets}

Target words: {task.target_words}
"""
            )
        ]
    ).content

    return {"sections": [(task.id, section)]}


# ============================================================
# Merge
# ============================================================

def merge_content(state: State):

    plan = state["plan"]

    ordered = [x[1] for x in sorted(state["sections"])]

    body = "\n\n".join(ordered)

    merged = f"# {plan.blog_title}\n\n{body}"

    return {"merged_md": merged}


# ============================================================
# Image Planner
# ============================================================

IMAGE_SYSTEM = """
Decide if images are required.
Maximum 3 images.
"""

def decide_images(state: State):

    planner = llm.with_structured_output(GlobalImagePlan)

    result = planner.invoke(
        [
            SystemMessage(content=IMAGE_SYSTEM),
            HumanMessage(content=state["merged_md"])
        ]
    )

    return {
        "md_with_placeholders": result.md_with_placeholders,
        "image_specs": [x.model_dump() for x in result.images]
    }


# ============================================================
# Final
# ============================================================

def generate_and_place_images(state: State):

    md = state.get("md_with_placeholders") or state["merged_md"]

    filename = "blog.md"

    Path(filename).write_text(md)

    return {"final": md}


# ============================================================
# Reducer Graph
# ============================================================

reducer = StateGraph(State)

reducer.add_node("merge_content", merge_content)
reducer.add_node("decide_images", decide_images)
reducer.add_node("generate_and_place_images", generate_and_place_images)

reducer.add_edge(START, "merge_content")
reducer.add_edge("merge_content", "decide_images")
reducer.add_edge("decide_images", "generate_and_place_images")
reducer.add_edge("generate_and_place_images", END)

reducer_graph = reducer.compile()


# ============================================================
# Main Graph
# ============================================================

g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_graph)

g.add_edge(START, "router")

g.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator"
    }
)

g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])

g.add_edge("worker", "reducer")

g.add_edge("reducer", END)

app = g.compile()


