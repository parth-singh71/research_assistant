from typing import Literal
from pydantic import BaseModel


class BriefRequest(BaseModel):
    topic: str
    depth: Literal[1, 2, 3] = 2
    user_id: str
    follow_up: bool = False
