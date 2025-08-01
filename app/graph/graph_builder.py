import os
import datetime
from dotenv import load_dotenv
from typing import Literal, List

from newspaper import Article
from tavily import TavilyClient

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.schemas import (
    FinalBrief,
    PlanningResponse,
    ResearchGraphState,
    ResearchStep,
    SearchResult,
    SourceSummary,
    WebSearchResult,
)
from app.db.models import Brief

# In-memory state saver for LangGraph (checkpointing during dev)
memory = InMemorySaver()

load_dotenv()
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))


async def get_all_briefs_for_user(user_id: str):
    """Fetch all previously saved briefs for the given user."""
    return await Brief.filter(user_id=user_id).order_by("created_at")


# === Node 1: context_summarization ===
async def context_summarization(
    state: ResearchGraphState,
) -> Command[Literal["planning_node"]]:
    """
    Loads prior briefs for the user and summarizes them to provide context
    for a follow-up research task.
    """
    if state.get("follow_up", False):
        user_briefs = await get_all_briefs_for_user(state["user_id"])
        if user_briefs:
            all_summaries = "\n\n".join(
                [
                    brief.brief_json.get("compiled_summary", "")
                    for brief in user_briefs
                    if brief.brief_json.get("compiled_summary")
                ]
            )

            llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
            prompt = ChatPromptTemplate(
                [
                    (
                        "system",
                        "You are a helpful assistant that summarizes a user's research history.",
                    ),
                    (
                        "user",
                        "Given the following previous research brief summaries, generate a concise summary that can be used as prior context:\n\n{summaries}",
                    ),
                ]
            )
            chain = prompt | llm | StrOutputParser()
            context_summary = chain.invoke({"summaries": all_summaries})
            return Command(
                update={"prior_context": context_summary}, goto="planning_node"
            )

    # No prior context needed; reset intermediate state
    return Command(
        update={
            "prior_context": None,
            "plan": None,
            "search_results": None,
            "fetched_content": None,
            "source_summaries": None,
            "final_brief": None,
        },
        goto="planning_node",
    )


# === Node 2: planning_node ===
def planning_node(state: ResearchGraphState) -> Command[Literal["search_node"]]:
    """
    Uses LLM to break the research topic into logical steps depending on the depth.
    """
    topic = state.get("topic")
    depth = state.get("depth")

    llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
    planning_llm = llm.with_structured_output(PlanningResponse)
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a research assistant.\n\n"
                "Your job is to break down a topic into key research steps.\n\n"
                "The `depth` parameter defines how detailed the steps should be:\n"
                "- Depth 1 = high-level overview\n"
                "- Depth 2 = moderate detail\n"
                "- Depth 3 = deep, analytical, and technical focus\n\n"
                "You also have access to user's past research summary if available",
            ),
            (
                "user",
                "Topic: {topic}\nDepth: {depth}\nPast Research Summary: {previous_summary}",
            ),
        ]
    )
    chain = prompt | planning_llm
    planning_resp: PlanningResponse = chain.invoke(
        {
            "topic": topic,
            "depth": depth,
            "previous_summary": state.get("prior_context", ""),
        }
    )

    return Command(update={"plan": planning_resp.research_steps}, goto="search_node")


# === Node 3: search_node ===
def search_node(state: ResearchGraphState) -> Command[Literal["content_fetching_node"]]:
    """
    Generates search queries for each step and fetches relevant URLs using Tavily API.
    Filters results by relevance.
    """
    search_results: List[WebSearchResult] = []
    research_steps: List[ResearchStep] = state.get("plan", [])

    for step in research_steps:
        query = generate_query_for_web_search(state.get("topic"), step)
        max_allowed_results = os.getenv("MAX_SEARCH_RESULTS")
        res = tavily_client.search(
            query, max_results=max_allowed_results if max_allowed_results else 1
        )

        sub_searches: List[SearchResult] = []
        if "results" in res:
            for sr in res["results"]:
                if sr.get("score", 0.0) >= 0.5:
                    sub_searches.append(
                        SearchResult(
                            title=sr["title"],
                            url=sr["url"],
                            content=sr["content"],
                            relevance_score=sr["score"],
                        )
                    )
        if len(sub_searches) > 0:
            search_results.append(
                WebSearchResult(
                    query=res["query"],
                    answer=res["answer"],
                    results=sub_searches,
                )
            )

    return Command(
        update={"search_results": search_results}, goto="content_fetching_node"
    )


def generate_query_for_web_search(topic: str, research_step: ResearchStep) -> str:
    """LLM-generated search query based on topic and step description."""
    llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a helpful assistant that creates concise, effective web search queries "
                "to retrieve relevant information about a topic.\n\n"
                "Given the main topic and a specific research step, "
                "your task is to generate a search engine query "
                "that can be used to find useful information for that step.",
            ),
            ("user", "Topic: {topic}\nResearch Step: {research_step}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": topic, "research_step": research_step.description})


# === Node 4: content_fetching_node ===
def content_fetching_node(
    state: ResearchGraphState,
) -> Command[Literal["summarization_node"]]:
    """
    Enhances search result content by scraping full article text when available.
    Replaces short previews with more complete content if found.
    """
    search_results = state.get("search_results", [])

    for i in range(len(search_results)):
        step_results = search_results[i]
        for j in range(len(step_results.results)):
            result = step_results.results[j]
            url = result.url
            new_content = scrape_web_content(url)
            # Replace only if content is significantly longer
            if new_content is not None and len(new_content.split(" ")) > len(
                result.content.split(" ")
            ):
                search_results[i].results[j].content = new_content

    return Command(update={"search_results": search_results}, goto="summarization_node")


def scrape_web_content(url):
    """Uses newspaper3k to extract clean article text from a given URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


# === Node 5: summarization_node ===
def summarization_node(state: ResearchGraphState) -> Command[Literal["synthesis_node"]]:
    """
    Summarizes the fetched content for each source using LLM.
    Stores structured summaries along with relevance scores.
    """
    search_results = state.get("search_results", [])
    summaries: List[SourceSummary] = []

    llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You summarize long-form content into concise, informative summaries.",
            ),
            ("user", "Summarize the following content:\n\n{content}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    for i in range(len(search_results)):
        step_results = search_results[i]
        for j in range(len(step_results.results)):
            result = step_results.results[j]
            try:
                summary = chain.invoke({"content": result.content})
                summary = summary.strip()
                search_results[i].results[j].summary = summary
                source_summary = SourceSummary(
                    source_url=result.url,
                    summary=summary,
                    relevance_score=result.relevance_score,
                )
                summaries.append(source_summary)
            except Exception as e:
                print(f"Failed to summarize {result.url}: {e}")
                continue

    return Command(
        update={"source_summaries": summaries, "search_results": search_results},
        goto="synthesis_node",
    )


# === Node 6: synthesis_node ===
def synthesis_node(state: ResearchGraphState) -> Command[Literal["postprocess_node"]]:
    """
    Synthesizes all individual summaries into one cohesive research brief.
    """
    summaries = state.get("source_summaries", [])
    summary_texts = "\n\n".join([s.summary for s in summaries])

    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a helpful assistant that synthesizes research summaries into a coherent brief. "
                "Using the following source summaries, generate a well-structured research brief",
            ),
            ("user", "Source Summaries:\n\n{summaries}"),
        ]
    )
    llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
    chain = prompt | llm | StrOutputParser()

    compiled_summary = chain.invoke({"summaries": summary_texts})
    return Command(
        update={"compiled_summary": compiled_summary.strip()}, goto="postprocess_node"
    )


# === Node 7: postprocess_node ===
async def postprocess_node(state: ResearchGraphState) -> Command[Literal["__end__"]]:
    """
    Creates FinalBrief object and saves it to the database for the user.
    """
    brief = FinalBrief(
        topic=state.get("topic"),
        steps=state.get("plan", []),
        summaries=state.get("source_summaries", []),
        references=state.get("search_results", []),
        compiled_summary=state.get("compiled_summary", ""),
    )
    await Brief.create(
        user_id=state["user_id"],
        topic=brief.topic,
        brief_json=brief.model_dump_json(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    return Command(update={"final_brief": brief}, goto="__end__")


# === Graph Builder ===
def build_graph(print_mermaid_graph: bool = False):
    """
    Constructs and returns the LangGraph pipeline for research brief generation.
    Optionally prints a Mermaid graph representation.
    """
    graph = StateGraph(ResearchGraphState)

    # Define all node functions
    graph.add_node("context_summarization", context_summarization)
    graph.add_node("planning_node", planning_node)
    graph.add_node("search_node", search_node)
    graph.add_node("content_fetching_node", content_fetching_node)
    graph.add_node("summarization_node", summarization_node)
    graph.add_node("synthesis_node", synthesis_node)
    graph.add_node("postprocess_node", postprocess_node)

    # Set entry and exit points
    graph.set_entry_point("context_summarization")

    app = graph.compile(checkpointer=memory)

    if print_mermaid_graph:
        print(app.get_graph().draw_mermaid())

    return app
