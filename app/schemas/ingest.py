from pydantic import BaseModel
from typing import List

class IngestRequest(BaseModel):
    paths: List[str]
