import json
import asyncio
import argparse
from app.utils.api_schemas import BriefRequest
from app.graph.graph_executor import execute_graph


async def init_and_run(topic, depth, follow_up, user_id):
    state = await execute_graph(
        BriefRequest(topic=topic, depth=depth, follow_up=follow_up, user_id=user_id)
    )
    print(json.dumps(state["final_brief"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--userid", required=True)
    args = parser.parse_args()

    followup = True if args.followup else False
    userid = args.userid if args.userid else "default"
    asyncio.run(init_and_run(args.topic, args.depth, followup, userid))


if __name__ == "__main__":
    main()
