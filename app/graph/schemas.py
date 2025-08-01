from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict


class ResearchStep(BaseModel):
    """
    Represents a single step in a structured research plan.
    Each step outlines a specific aspect or subtopic to explore under the main topic.
    """

    step_id: int = Field(
        ..., description="Step number in the research plan, starting from 1"
    )
    description: str = Field(
        ..., description="A short, clear description of the research step"
    )


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    summary: Optional[str] = None
    relevance_score: Optional[float] = None


class WebSearchResult(BaseModel):
    query: str
    answer: Optional[str]
    results: List[SearchResult]


class SourceSummary(BaseModel):
    source_url: str
    summary: str
    relevance_score: float


class FinalBrief(BaseModel):
    topic: str
    steps: List[ResearchStep]
    summaries: List[SourceSummary]
    compiled_summary: str
    references: List[WebSearchResult]


class PlanningResponse(BaseModel):
    """
    A structured response model that contains multiple research steps
    derived from the input topic and desired depth.
    """

    research_steps: List[ResearchStep] = Field(
        description="List of research steps to investigate the topic in structured order"
    )


# --- State Definition ---


class ResearchGraphState(TypedDict):
    topic: str
    user_id: str
    depth: int
    follow_up: bool
    prior_context: Optional[str] = None
    plan: Optional[List[ResearchStep]] = None
    search_results: Optional[List[WebSearchResult]] = None
    source_summaries: Optional[List[SourceSummary]] = None
    compiled_summary: Optional[str]
    final_brief: Optional[FinalBrief] = None
