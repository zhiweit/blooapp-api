from pydantic import BaseModel
from typing import Optional


class ImageRequest(BaseModel):
    base64_image: str


class ChatRequest(BaseModel):
    question: Optional[str] = None
    base64_image: Optional[str] = None
    thread_id: Optional[str] = None


class Question(BaseModel):
    question: str
