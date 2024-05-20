import asyncio
import sys
import uvicorn

from fastapi import FastAPI, APIRouter, HTTPException
from app.models.image_request import ImageRequest
from app.models.question import Question
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


@router.post("/vision/json")
async def vision_json(request: ImageRequest):
    base64_image = request.base64_image

    # region: Identifying the items in the image
    image_prompt = """
    I have an image containing items that I am unsure of whether they are recyclable. Please help me to identify the item(s) in the image. Return the answer as image_items where the number of items is according to the items you have identified.

    Example answer:
    image_items:
    <items identified in the image>
    
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

    Return the answer in JSON.
    Return the response in JSON format.
    Example of a valid response with single item identified that is recyclable:
    {{
        "item": {{
            "name": "Paper Milk Carton",
            "description": "The item in the image is a paper milk carton, specifically a 1-liter container for low-fat fresh milk.",
            "recyclable": 'true',
            "instructions": "In Singapore, paper milk cartons like this can be recycled, but it's important to prepare them correctly because they are often lined with plastic or aluminum. Here's how to recycle this item properly:

            1. **Empty the Carton**: Ensure the carton is completely empty of any milk.
            2. **Rinse the Carton**: Rinse it out to remove any milk residue, as residue can contaminate other recyclables.
            3. **Dry the Carton**: Allow the carton to dry to prevent mold growth in the recycling bin.
            4. **Flatten the Carton**: Flatten the carton to save space in your recycling bin and facilitate easier transportation and processing.
            5. **Recycling Bin**: Place the clean, dry, and flattened carton in the recycling bin designated for paper or comingled recyclables, depending on your local recycling guidelines.

            By following these steps, you help ensure that the carton is recycled efficiently and does not contaminate other recyclable materials."
        }},
        "other_items": []
    }}

    Example of a valid response with single item identified that is not recyclable:
    {{
        "item": {{
            "name": "Sheet Mask",
            "description": "The item in the image is a sheet mask, typically used for skincare. These masks are generally made from a lightweight fabric-like material that is infused with various skincare serums.",
            "recyclable": 'false',
            "instructions": "In Singapore, sheet masks are not recyclable through regular municipal recycling programs due to their composition and contamination with cosmetic products. Here's what you can do:

            1. **Dispose of Properly**: Since the sheet mask itself and the serum it contains can contaminate other recyclables, it should be disposed of in the general waste bin.
            2. **Check the Packaging**: If the sheet mask came in a separate packaging, such as a plastic wrapper or paper box, check those for recycling symbols. Clean and dry them before recycling if they meet the local recycling guidelines.
            3. **Reduce Waste**: To minimize waste, consider using reusable face masks or those made from biodegradable materials if available.
            4. **Special Recycling Programs**: Occasionally, cosmetic brands or stores offer recycling programs specifically for beauty product packaging. Inquire at the place of purchase or directly with the brand to see if they provide such a service.

            Always refer to the latest guidelines from the National Environment Agency (NEA) of Singapore for the most up-to-date information on waste management practices."
        }},
        "other_items": []
    }}

    Example of a valid response with single item identified that is partially recyclable:
    {{
        "item": {{
            "name": "Cosmetic container",
            "description": "The item in the image is a cosmetic container, specifically for a facial cream with a pump dispenser.",
            "recyclable": 'partial',
            "instructions": "In Singapore, cosmetic containers made of plastic can be recycled, but you should separate the components because they often include different materials. Here's how to recycle this item properly:

            1. **Empty the Container**: Make sure the container is completely empty of any product.
            2. **Separate Components**: Detach the pump dispenser from the bottle, as the pump often contains metal springs and other non-recyclable components.
            3. **Clean the Container**: Rinse the plastic container to remove any residual product.
            4. **Dry the Container**: Allow it to air dry to avoid moisture in the recycling bin.
            5. **Discard Non-recyclable Parts**: Dispose of the pump in general waste, unless your local recycling program specifies that it can be recycled.
            6. **Recycle the Plastic Part**: Place the clean and dry plastic container in the recycling bin designated for plastics.

            It's important to follow these steps to ensure that the recyclable parts are processed correctly and to reduce contamination in the recycling stream."
        }},
        "other_items": []
    }}

    Example of a valid response with multiple items in the image:
    {{
        "item": {{
            "name": "Shirt",
            "description": "The item in the image is a white long-sleeve shirt hanging on a plastic hanger.",
            "recyclable": 'false',
            "instructions": "If the shirt is in good condition, it is best to donate it to a charity or second-hand store. If it is worn out, some recycling programs accept textiles where they can be turned into industrial rags, insulation, or other textile byproducts. Check with local textile recycling facilities or drop-off points.
        }},
        "other_items": [
            {{
                "name": "Plastic Hanger",
                "description": "The item in the image is a plastic hanger.",
                "recyclable": 'false',
                "instructions": "Plastic hangers are typically not recyclable through curbside recycling programs due to their size, shape, and the mixed plastics they are often made from. Consider donating usable hangers to thrift stores or returning them to dry cleaners. If they are broken, they should be disposed of in the general waste unless a specific recycling option is available."
            }}
        ]
    }}
}}

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
        | qa_model_json_output
        # | JsonOutputParser(pydantic_object=ImageResponse)
    )

    res = chain_text_stream.invoke({"image_prompt": image_prompt})
    # print(res)
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
