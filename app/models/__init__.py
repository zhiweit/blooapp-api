from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from enum import Enum


class Item(BaseModel):
    material: str = Field(description="Material of the item")
    item: str = Field(description="Name of the item")
    recyclable: bool = Field(description="Whether the item can be recycled or not")
    instructions: str = Field(
        description="Recycling instructions for each item if the item is recyclable, or instructions to dispose of the item if the item is not recyclable"
    )


class ItemNames(BaseModel):
    items: List[str] = Field(description="List of item names")
