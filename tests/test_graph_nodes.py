import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.graph.graph_builder import (
    context_summarization,
    planning_node,
    search_node,
    content_fetching_node,
    summarization_node,
    synthesis_node,
    postprocess_node,
)
from app.graph.schemas import (
    ResearchGraphState,
    ResearchStep,
    SearchResult,
    WebSearchResult,
)


@pytest.fixture
def basic_state():
    return {
        "topic": "Future of AI",
        "depth": 1,
        "follow_up": False,
        "user_id": "test_user",
    }


class FakeChain:
    def invoke(self, input):
        return "Prior Context Summary"


@patch("app.graph.graph_builder.get_all_briefs_for_user", new_callable=AsyncMock)
@patch("app.graph.graph_builder.init_chat_model")
@pytest.mark.asyncio
async def test_context_summarization_followup(
    mock_llm_init, mock_get_briefs, basic_state
):
    basic_state["follow_up"] = True
    mock_get_briefs.return_value = [
        MagicMock(brief_json={"compiled_summary": "Summary 1"}),
        MagicMock(brief_json={"compiled_summary": "Summary 2"}),
    ]
    mock_llm_init.return_value.__or__.return_value.__or__.return_value = FakeChain()

    result = await context_summarization(basic_state)
    assert result.goto == "planning_node"
    assert result.update["prior_context"] == "Prior Context Summary"


def test_planning_node(mocker, basic_state):
    class FakeLLM:
        def with_structured_output(self):
            return self

        def invoke(self, input):
            return MagicMock(
                research_steps=[ResearchStep(step_id=1, description="Step 1")]
            )

    mocker.patch("app.graph.graph_builder.init_chat_model", return_value=FakeLLM())

    result = planning_node(basic_state)
    assert result.goto == "search_node"
    assert len(result.update["plan"]) == 1


def test_search_node(mocker):
    state = {
        "topic": "AI in Medicine",
        "plan": [ResearchStep(step_id=1, description="Impact of AI in diagnostics")],
    }

    mocker.patch(
        "app.graph.graph_builder.generate_query_for_web_search",
        return_value="test query",
    )
    mock_search = MagicMock()
    mock_search.search.return_value = {
        "query": "test query",
        "answer": "summary",
        "results": [
            {
                "title": "Article 1",
                "url": "http://a.com",
                "content": "text",
                "score": 0.9,
            }
        ],
    }
    mocker.patch("app.graph.graph_builder.tavily_client", mock_search)

    result = search_node(state)
    assert result.goto == "content_fetching_node"
    assert len(result.update["search_results"]) == 1


def test_content_fetching_node(mocker):
    state = {
        "search_results": [
            WebSearchResult(
                query="test",
                answer="answer",
                results=[
                    SearchResult(
                        title="Title",
                        url="http://a.com",
                        content="dummy content",
                        relevance_score=0.8,
                    )
                ],
            )
        ]
    }
    mocker.patch(
        "app.graph.graph_builder.scrape_web_content",
        return_value="Much longer article content",
    )

    result = content_fetching_node(state)
    assert result.goto == "summarization_node"
    assert "search_results" in result.update


def test_summarization_node(mocker):
    class FakeChain:
        def invoke(self, input):
            return "Summarized text"

    mock_llm = MagicMock()
    mock_llm.__or__.return_value.__or__.return_value = FakeChain()
    mocker.patch("app.graph.graph_builder.init_chat_model", return_value=mock_llm)

    state = {
        "search_results": [
            WebSearchResult(
                query="query",
                answer="answer",
                results=[
                    SearchResult(
                        title="Title",
                        url="http://test.com",
                        content="Content to summarize.",
                        relevance_score=0.7,
                    )
                ],
            )
        ]
    }

    result = summarization_node(state)
    assert result.goto == "synthesis_node"
    assert len(result.update["source_summaries"]) == 1


def test_synthesis_node(mocker):
    class FakeChain:
        def invoke(self, input):
            return "Compiled Research Summary"

    mock_llm = MagicMock()
    mock_llm.__or__.return_value.__or__.return_value = FakeChain()
    mocker.patch("app.graph.graph_builder.init_chat_model", return_value=mock_llm)

    state = {
        "source_summaries": [
            MagicMock(summary="First source."),
            MagicMock(summary="Second source."),
        ]
    }

    result = synthesis_node(state)
    assert result.goto == "postprocess_node"
    assert "compiled_summary" in result.update


@patch("app.graph.graph_builder.Brief.create", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_postprocess_node(mock_create):
    state = {
        "user_id": "u123",
        "topic": "Quantum Computing",
        "plan": [ResearchStep(step_id=1, description="Intro")],
        "source_summaries": [],
        "search_results": [],
        "compiled_summary": "Final Summary",
    }

    result = await postprocess_node(state)
    assert result.goto == "__end__"
    assert "final_brief" in result.update
