import operator
from typing import Annotated, List, Optional, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


from langgraph.graph import StateGraph, START, END


class State(BaseModel):
    topic: str
    joke: str = Field(default="")


LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate(
    template="""Write me a funny story about {topic}. The story should be 3 paragraphs long. Each paragraph has at least five sentences.""",
    input_variables=["topic"],
)

generate_joke_chain = prompt | LLM


async def generate_joke(state: State, config: RunnableConfig):
    topic = state.topic
    res = await generate_joke_chain.ainvoke({"topic": topic}, config=config)
    return {"joke": res}


print("Building graph...")
graph_builder = StateGraph(State)
graph_builder.add_node("generate_joke", generate_joke)
graph_builder.add_edge(START, "generate_joke")
graph_builder.add_edge("generate_joke", END)

graph = graph_builder.compile()
