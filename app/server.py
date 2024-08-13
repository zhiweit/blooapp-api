import asyncio
import sys
from typing import List, Optional
import uuid
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.request_body import ChatRequest, ImageRequest, Question
from app.models import Item, ItemNames
from app.dependencies import (
    vision_model,
    qa_model_item_output,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from fastapi.responses import StreamingResponse
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import PromptTemplate
import logging
from app.firestore import collection
from app.models import ItemNames
from google.cloud.firestore_v1.base_query import FieldFilter
from app.agents.recycling.main import graph_builder, State, AsyncFirestoreDBSaver
from app.agents.jokes.joke_agent import State as JokeState
import time
import json


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncFirestoreDBSaver.from_conn_info(
        checkpoints_collection_name="checkpoints",
        checkpoint_writes_collection_name="checkpoint_writes",
    ) as checkpointer:
        global graph
        graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("Graph compiled")
    yield
    logger.info("Graph cleaned up")
    # Clean up


app = FastAPI(lifespan=lifespan)
origins = [
    "http://localhost:3000",  # uncomment for local dev with frontend
    "https://blooapp.vercel.app",
    "https://blooapp-zhiweits-projects.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")
chat_history_store = {}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]


@router.get("/health")
async def health_check():
    logger.info("hello world")
    return {"message": "hello world"}
    # return RedirectResponse("/docs")


async def stream_event(event_name, payload, message_id, logger, log=True):
    event_data_json = {"messageId": message_id, "payload": payload}
    event_name = "event: {name}".format(name=event_name)
    event_data = "data: {data}".format(data=json.dumps(event_data_json))
    if log:
        logger.info(payload)
    return f"{event_name}\n{event_data}\n\n"


async def stream_chat(state: State, config: Optional[RunnableConfig] = None):
    generated_ans = []
    # generate a unique message id
    message_id = str(uuid.uuid4())
    async for event in graph.astream_events(state.dict(), config, version="v2"):
        # event_name = "event: {name}"
        # event_data = "data: {data}"
        # event_data_json = {"messageId": message_id, "type": "", "content": ""}
        kind = event["event"]
        langgraph_node = event["metadata"].get("langgraph_node", None)
        name = event["name"]

        if kind == "on_chain_start":
            if name == "LangGraph":
                yield await stream_event(
                    "start", "Langgraph start...", message_id, logger
                )

            elif (
                name == "identify_image_items"
                and langgraph_node == "identify_image_items"
            ):
                yield await stream_event(
                    "thinking", "🔍 Identifying items from image...", message_id, logger
                )

            elif (
                name == "rephrase_question_based_on_image_items"
                and langgraph_node == "rephrase_question_based_on_image_items"
            ):
                yield await stream_event(
                    "thinking",
                    "♻️ Rephrasing question based on image items...",
                    message_id,
                    logger,
                )

            elif (
                name == "rephrase_question_based_on_chat_history"
                and langgraph_node == "rephrase_question_based_on_chat_history"
            ):
                yield await stream_event(
                    "thinking",
                    "♻️ Rephrasing question based on chat history...",
                    message_id,
                    logger,
                )

            elif name == "retrieve_docs" and langgraph_node == "retrieve_docs":
                yield await stream_event(
                    "thinking", "📚 Retrieving documents...", message_id, logger
                )

            elif (
                name == "grade_retrieved_docs"
                and langgraph_node == "grade_retrieved_docs"
            ):
                yield await stream_event(
                    "thinking", "📝 Grading retrieved documents...", message_id, logger
                )

            elif (
                name == "perform_web_search" and langgraph_node == "perform_web_search"
            ):
                yield await stream_event(
                    "thinking", "🔍 Performing web search...", message_id, logger
                )

            elif name == "generate_answer" and langgraph_node == "generate_answer":
                yield await stream_event(
                    "thinking", "💬 Generating answer...", message_id, logger
                )

            elif (
                name == "generate_answer_from_llm"
                and langgraph_node == "generate_answer_from_llm"
            ):
                yield await stream_event(
                    "thinking",
                    "🤖 Generating answer from external sources (LLM)...",
                    message_id,
                    logger,
                )

        elif kind == "on_chain_end":
            if (
                name == "identify_image_items"
                and langgraph_node == "identify_image_items"
            ):
                output = event["data"]["output"].get("items", None)
                payload = f"✓ Identified items from image: {','.join(output) if len(output) > 0 else 'None'}"
                yield await stream_event("thinking", payload, message_id, logger)

            elif (
                name == "rephrase_question_based_on_chat_history"
                and langgraph_node == "rephrase_question_based_on_chat_history"
            ):
                rephrased_question = event["data"]["output"].get(
                    "rephrased_question", None
                )
                payload = f"✓ Rephrased question: {rephrased_question}"
                yield await stream_event("thinking", payload, message_id, logger)

            elif name == "retrieve_docs" and langgraph_node == "retrieve_docs":
                payload = f"✓ Retrieved documents"
                yield await stream_event("thinking", payload, message_id, logger)

            elif (
                name == "grade_retrieved_docs"
                and langgraph_node == "grade_retrieved_docs"
            ):
                output = event["data"]["output"].get("is_retrieved_docs_relevant", None)
                payload = f"✓ Graded retrieved documents. {'Documents are relevant to question' if output else 'Documents are not relevant to question'}"
                yield await stream_event("thinking", payload, message_id, logger)

            elif (
                name == "perform_web_search" and langgraph_node == "perform_web_search"
            ):
                payload = f"✓ Performed web search"
                yield await stream_event("thinking", payload, message_id, logger)

            elif (
                name == "generate_answer" and langgraph_node == "generate_answer"
            ) or (
                name == "generate_answer_from_llm"
                and langgraph_node == "generate_answer_from_llm"
            ):
                generated_ans = event["data"]["output"].get("generated_answer", None)
                logger.info(f"\n\nGenerated answer: {generated_ans}")
                yield await stream_event(
                    "generated_answer", generated_ans, message_id, logger
                )

            elif name == "LangGraph":
                thread_id = config["configurable"].get("thread_id", None)
                logger.info(f"Langgraph end. Thread ID: {thread_id}")
                yield await stream_event("thread_id", thread_id, message_id, logger)

        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if (
                langgraph_node == "generate_answer"
                or langgraph_node == "generate_answer_from_llm"
            ):
                if content:
                    # Empty content in the context of OpenAI or Anthropic usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    generated_ans.append(content)
                    payload = "".join(generated_ans)
                    yield await stream_event(
                        "generate_answer", payload, message_id, logger, log=False
                    )


@router.post("/chat")
async def chat(request: ChatRequest):
    state = State()
    thread_id = request.thread_id if request.thread_id else str(uuid.uuid4())
    config = {
        "configurable": {
            "max_web_search_count": 5,
            "debug": False,
            "thread_id": thread_id,
        }
    }
    if request.question:
        state.user_question = request.question
    if request.base64_image:
        config["configurable"]["base64_image"] = request.base64_image

    return StreamingResponse(
        stream_chat(state, config),
        media_type="text/event-stream",
        headers={"Content-Encoding": "none"},
    )


async def streamer():
    # Sends an event every second with data: "Message {i}"
    for i in range(10):
        event_name = "event: stream_event"
        event_data = f"data: Message {i}"
        yield f"{event_name}\n{event_data}\n\n"
        await asyncio.sleep(1)


@router.post("/stream-test")
async def streaming_test():
    return StreamingResponse(
        streamer(), media_type="text/event-stream", headers={"Content-Encoding": "none"}
    )


# Testing only
@router.post("/joke")
async def joke(request: Question):
    topic = request.question or "monkey"
    state = JokeState(topic=topic)
    try:
        # Define an async generator to yield results
        async def stream_results():
            async for event in graph.astream_events(state.dict(), version="v2"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # Empty content in the context of OpenAI or Anthropic usually means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content
                        print(content, end="|")
                        event_str = "event: stream_event"
                        data_str = f"data: {content}"
                        yield f"{event_str}\n{data_str}\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            headers={"Content-Encoding": "none"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)
if __name__ == "__main__":
    dev_mode = "dev" in sys.argv
    print(f"Running in {'development' if dev_mode else 'production'} mode")
    uvicorn.run("app.server:app", host="0.0.0.0", port=8080, reload=dev_mode)
