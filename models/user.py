from pydantic import BaseModel
from typing import List

class User(BaseModel):
    user_id: str
    name: str
    age: int
    embedding: List[float]
