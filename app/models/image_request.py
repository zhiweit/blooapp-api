from pydantic import BaseModel


class ImageRequest(BaseModel):
    base64_image: str


class Question(BaseModel):
    question: str
