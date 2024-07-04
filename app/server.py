import asyncio
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.image_request import ImageRequest
from app.models import Item, ItemNames
from app.dependencies import (
    vision_model,
    qa_model,
    retriever,
    qa_model_item_output,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi.responses import StreamingResponse
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
import logging
from app.firestore import collection
from app.models.question import Question
from app.models import ItemNames
from google.cloud.firestore_v1.base_query import FieldFilter


@asynccontextmanager
async def lifespan(app: FastAPI):
    global NEA_ITEM_NAMES, NEA_ITEM_NAMES_SET
    NEA_ITEM_NAMES = [doc.to_dict()["item"] for doc in collection.stream()]
    NEA_ITEM_NAMES_SET = set(NEA_ITEM_NAMES)
    logger.info(f"{len(NEA_ITEM_NAMES_SET)} NEA item names loaded")
    yield
    # Clean up
    NEA_ITEM_NAMES_SET.clear()
    logger.info("NEA item names cleared from memory")


app = FastAPI(lifespan=lifespan)
origins = [
    # "http://localhost:3000",  # uncomment for local dev with frontend
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


@router.get("/")
async def redirect_root_to_docs():
    logger.info("hello world")
    return {"message": "hello world"}
    # return RedirectResponse("/docs")


@router.post("/vision")
async def get_item_names(request: ImageRequest):
    base64_image = request.base64_image

    # region: Identifying the items in the image
    image_prompt = """
    I have an image containing items that I am unsure of whether they are recyclable. Please help me to identify the item(s) in the image.
    For each of the items, find the best or closest matching item from the following NEA_ITEM_NAMES, and return it. If there is no best match for the item, return the item according to the name that you have identified.
    The number of items returned should be the same as the number of items identified in the image.

    NEA_ITEM_NAMES: \n\n {NEA_ITEM_NAMES} \n\n

    Return the answer as JSON output according to the following schema:
    {schema}

    """

    image_prompt = image_prompt.format(
        NEA_ITEM_NAMES=NEA_ITEM_NAMES, schema=ItemNames.schema_json()
    )

    image_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert on answering questions briefly and accurately about recycling in Singapore. Users may send you images of items to check if the items can be recycled, and your task is to correctly identify what are the items in the image, and provide the recycling instructions of the items.",
            ),
            (
                "human",
                [
                    {"type": "text", "text": "{image_prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            ),
        ]
    )

    # endregion

    vision_model_json_output = vision_model.with_structured_output(schema=ItemNames)

    vision_chain = image_prompt_template | vision_model_json_output

    item_names = vision_chain.invoke({"image_prompt": image_prompt}).items
    item_names = [item.lower() for item in item_names]

    res = {}
    logger.info(f"Item names: {item_names}")

    # look up db to get recycling instructions
    docs = collection.where(filter=FieldFilter("item", "in", item_names)).stream()

    res["from_database"] = [doc.to_dict() for doc in docs]
    res["from_llm"] = []

    # for the item names that are not in the NEA list, prompt llm on the recycling instructions
    unmapped_item_names = [i for i in item_names if i not in NEA_ITEM_NAMES_SET]
    if len(unmapped_item_names) > 0:
        prompt = PromptTemplate.from_template(
            """
        You are an expert on answering questions briefly and accurately about recycling in Singapore. You help to answer users' questions on whether the items are recyclable, and provide instructions on how to properly recycle or dispose of them.
        Answer the following question. Return the answer in JSON format according to the following schema:
        Schema: \n\n {schema} \n\n

        Question:
        \n\n {question} \n\n

        """
        )

        chain = prompt | qa_model_item_output

        unmapped_item_res = chain.batch(
            [
                {
                    "question": f"Is {item} recyclable in Singapore? If so, provide the recycling instructions for it. If not, provide the instructions to properly dispose of it.",
                    "schema": Item.schema_json(),
                }
                for item in unmapped_item_names
            ]
        )

        res["from_llm"] = unmapped_item_res

    logger.info(f"res: {res}")
    return res


@router.post("/vision/stream")
async def vision_stream(request: ImageRequest, session_id: str = "123"):
    base64_image = request.base64_image

    # region: Identifying the items in the image
    image_prompt = """
    I have an image containing items that I am unsure of whether they are recyclable. Please help me to identify the item(s) in the image. Return the answer as image_items where the number of items is according to the items you have identified.
    """

    image_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert on answering questions briefly and accurately about recycling in Singapore. Your name is Bloo. Users may send you images of items to check if the items can be recycled, and your task is to correctly identify what are the items in the image, and help answer users' questions on whether the items are recyclable or not in a helpful, correct and concise manner.",
            ),
            (
                "human",
                [
                    {"type": "text", "text": "{image_prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            ),
        ]
    )

    # endregion

    # region: Answering question on what items are recyclable, and providing instructions
    recycling_question = """
    For each of the image_items, is the item recyclable in Singapore? If so, provide the recycling instructions. If the item is not recyclable. If the item is not recyclable, answer why the item(s) are not recyclable and how to properly dispose it.

    """

    template = """Answer the question referring to the following context. 
    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # endregion

    chain_text_stream = (
        image_prompt_template
        | vision_model
        | StrOutputParser()
        | RunnableParallel(
            {
                "context": retriever,
                "question": RunnableLambda(
                    lambda vision_model_output: f"{vision_model_output}\n {recycling_question}"
                ),
            }
        )
        | prompt
        | qa_model
        | StrOutputParser()
    )

    with_message_history = RunnableWithMessageHistory(
        chain_text_stream,
        get_session_history,
        input_messages_key="image_prompt",
        history_messages_key="history",
    )

    try:
        # Define an async generator to yield results
        async def stream_results():
            async for chunk in with_message_history.astream(
                {"image_prompt": image_prompt},
                config={"configurable": {"session_id": session_id}},
            ):
                print(chunk, end="", flush=True)  # Optional: for debugging
                data_to_send = f"data:{chunk}\n\n"
                # Yield the data to the client
                yield data_to_send

        return StreamingResponse(stream_results(), media_type="text/event-stream")
    except Exception as e:
        # Log the exception before raising HTTPException
        logger.error(f"Error during streaming: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/vision/chat")
async def vision_stream_chat(session_id: str, question: Question):
    logger.info(f"session_id: {session_id}")
    logger.info(f"chat_history_store: {chat_history_store}", end="\n\n")
    question = question.question

    # region: Chat history
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # retriever that retrieves the documents based on prompt question and chat history
    history_aware_retriever = create_history_aware_retriever(
        qa_model, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """You are an expert on answering questions briefly and accurately about recycling in Singapore. \
    Your name is Bloo. You help answer users' recycling queries in a helpful, correct and concise manner. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(qa_model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # endregion

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    try:
        # Define an async generator to yield results
        async def stream_results():
            async for chunk in conversational_rag_chain.astream(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            ):
                # Assuming the chunk is a dictionary and the answer is under the key 'answer'
                if "answer" in chunk:
                    answer = chunk["answer"]
                    print(answer, end="", flush=True)  # Optional: for debugging
                    yield f"data:{answer}\n\n"
                # print(chunk, end="|", flush=True)
                # yield f"data:{chunk}\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)
if __name__ == "__main__":
    dev_mode = "dev" in sys.argv
    print(f"Running in {'development' if dev_mode else 'production'} mode")
    uvicorn.run("app.server:app", host="0.0.0.0", port=8080, reload=dev_mode)
