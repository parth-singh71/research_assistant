"""
This script initializes the database (Tortoise ORM), builds the research graph,
executes it with a given input, and prints the results step-by-step using `rich`.
"""

import asyncio
from rich import print as rprint
from rich.pretty import Pretty
from rich.panel import Panel
from rich.console import Console

from tortoise import Tortoise

from app.graph.graph_builder import build_graph
from app.graph.schemas import ResearchGraphState
from app.db.config import TORTOISE_ORM
from app.utils.api_schemas import BriefRequest

console = Console()


async def execute_graph(graph_input: BriefRequest):
    """
    Executes the research assistant LangGraph pipeline end-to-end
    for a single topic request.

    Args:
        graph_input (BriefRequest): Input schema containing topic, depth, user_id, etc.

    Returns:
        dict: Final state snapshot after graph execution
    """

    # Initialize ORM and apply schema if needed (dev mode only)
    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas()

    # Build the LangGraph pipeline with nodes
    graph = build_graph()

    initial_state: ResearchGraphState = {
        "topic": graph_input.topic,
        "depth": graph_input.depth,
        "follow_up": graph_input.follow_up,
        "user_id": graph_input.user_id,
    }

    config = {
        "configurable": {"thread_id": graph_input.user_id}
    }  # Thread ID for checkpointer

    # Stream node-level updates from LangGraph and print them
    events = graph.astream(initial_state, config, stream_mode="updates")
    async for event in events:
        step_name = next(iter(event))  # Current step node name
        data = event[step_name]  # Output payload from that node

        console.rule(f"[bold green]Step {step_name}[/bold green]")
        rprint(
            Panel.fit(Pretty(data), title=f"Payload: {step_name}", border_style="blue")
        )

    # Cleanup DB connection
    await Tortoise.close_connections()

    state_snapshot = graph.get_state(config)
    return state_snapshot.values


if __name__ == "__main__":
    # Example usage if called as a script (for testing)
    asyncio.run(execute_graph())
