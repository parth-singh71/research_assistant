from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel
from app.graph.schemas import FinalBrief
from app.utils.api_schemas import BriefRequest
from app.graph.graph_executor import execute_graph

router = APIRouter()


@router.post("/brief", response_model=FinalBrief)
async def generate_brief(payload: BriefRequest):
    state = await execute_graph(payload)
    return state["final_brief"]
