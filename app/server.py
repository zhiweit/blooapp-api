import asyncio
import sys
import uvicorn

from fastapi import FastAPI, APIRouter, HTTPException
from app.models.image_request import ImageRequest
from app.models.question import Question
from app.models.image_response import Items
from app.dependencies import (
    vision_model,
    qa_model,
    retriever,
    qa_model_json_output,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi.responses import StreamingResponse
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.templates import NEA_ITEM_NAMES

app = FastAPI()
router = APIRouter(prefix="/api")
chat_history_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]


@router.get("/")
async def redirect_root_to_docs():
    return {"message": "hello world"}
    # return RedirectResponse("/docs")


@router.post("/vision")
async def get_item_names(request: ImageRequest):
    base64_image = request.base64_image

    # region: Identifying the items in the image
    image_prompt = """
    I have an image containing items that I am unsure of whether they are recyclable. Please help me to identify the item(s) in the image. 
    Map the items you have identified to the following NEA_ITEM_NAMES: {NEA_ITEM_NAMES}

    Return the item as "Other" if the item is not in the list of NEA_ITEM_NAMES.
    
    Return the answer as JSON output according to the following schema:
    {{
        "items": ['item1', 'item2', ...]
    }}

    """

    image_prompt = image_prompt.format(NEA_ITEM_NAMES=NEA_ITEM_NAMES)

    image_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert on answering questions briefly and accurately about recycling in Singapore. Your name is Bloo. Users may send you images of items to check if the items can be recycled, and your task is to correctly identify what are the items in the image.",
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

    vision_model_json_output = vision_model.with_structured_output(schema=Items)

    chain_text_stream = image_prompt_template | vision_model_json_output

    res = chain_text_stream.invoke({"image_prompt": image_prompt})
    print(res)
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
                print(chunk, end="", flush=True)
                yield f"data:{chunk}\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/vision/chat")
async def vision_stream_chat(session_id: str, question: Question):
    print(f"session_id: {session_id}")
    print(f"chat_history_store: {chat_history_store}", end="\n\n")
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
