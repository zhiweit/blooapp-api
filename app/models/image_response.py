from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from enum import Enum


class RecyclableStatus(Enum):
    TRUE = "true"
    FALSE = "false"
    PARTIAL = "partial"


class Item(BaseModel):
    name: str = Field(description="item identified by the model")
    description: str = Field(description="description of the item")
    recyclable: RecyclableStatus = Field(description="whether the item is recyclable")
    instructions: str = Field(description="recycling instructions")


class ImageResponse(BaseModel):
    item: Item = Field(description="primary item identified by the model")
    other_items: List[Item] = Field(
        description="list of other possible items identified by the model"
    )
