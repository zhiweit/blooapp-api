from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from enum import Enum


class Item(BaseModel):
    name: str = Field(description="item identified by the model")
    description: str = Field(description="description of the item")
    recyclable: bool = Field(description="whether the item is recyclable")
    instructions: str = Field(
        description="recycling instructions for each item if the item is recyclable, or instructions to dispose"
    )


class ImageResponse(BaseModel):
    item: Item = Field(description="primary item identified by the model")
    other_items: List[Item] = Field(
        description="list of other possible items identified by the model"
    )


class Items(BaseModel):
    items: List[str] = Field(description="List of item names")
