from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from typing import Annotated, List, Optional, Sequence, Literal
import operator
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.dependencies import (
    LLM,
    vision_model,
    hybrid_search,
    web_search_tool,
    upload_base64_image_to_gcs,
    extract_image_data,
    IMAGE_FOLDER,
    bucket_ref,
)
import uuid


# region: State
class State(BaseModel):
    user_question: Optional[str]
    rephrased_question: str = Field(default="")
    items: List[str] = Field(default_factory=list)
    retrieved_docs: List[Document] = Field(default_factory=list)
    is_retrieved_docs_relevant: bool = Field(default=False)
    web_search_count: int = Field(default=0)
    messages: Annotated[Sequence[BaseMessage], operator.add] = Field(
        default_factory=list
    )
    generated_answer: str = Field(default="")


# endregion: State


# region: is_image_present
async def is_image_present(state: State, config: RunnableConfig):
    """Check if the user has provided an image for downstream tasks"""

    base64_image = config["configurable"].get(
        "base64_image", None
    )  # pass base64 image in the config instead of state to avoid persisting the image in the db via the checkpoiner
    if base64_image is None:
        return "no_image"
    return "has_image"


# endregion: has image

# region: identify image items
VISION_MODEL_SYSTEM_MESSAGE = """
You are a helpful assistant that identifies items in an image, to be used for downstream tasks like answering recycling questions on the items.
"""


class ItemNames(BaseModel):
    items: List[str] = Field(description="Item names identified from the image")


IMAGE_PROMPT = """
I have an image containing items that I am unsure of whether they are recyclable. Please help me to identify the item(s) in the image. If there are brand names in the image, ignore the brand names. Return the answer in JSON \n\n
The number of items in the `items` field in the JSON answer should be the same as the number of items you have identified in the image.
"""


image_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            VISION_MODEL_SYSTEM_MESSAGE,
        ),
        (
            "human",
            [
                {"type": "text", "text": IMAGE_PROMPT},
                {
                    "type": "image_url",
                    # "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
                    "image_url": {"url": "{base64_image}"},
                },
            ],
        ),
    ]
)

identify_items_chain = image_prompt_template | vision_model.with_structured_output(
    schema=ItemNames
)


async def identify_image_items(state: State, config: RunnableConfig):
    """Extract items from the image and return the items in the state

    Keyword arguments:
    state -- State object containing the question and context
    config -- RunnableConfig object containing the configuration
    Return: State object with the image items with docs
    """

    base64_image = config["configurable"].get("base64_image", None)
    if base64_image is None:
        raise ValueError("base64_image is required")

    identify_items_chain = image_prompt_template | vision_model.with_structured_output(
        schema=ItemNames
    )
    image_items = await identify_items_chain.ainvoke(
        {"base64_image": base64_image}, config
    )
    image_items = [item for item in image_items.items]
    return {"items": image_items}


# endregion: identify image items

# region: rephrase question based on image items
# TODO: Update this prompt
REPHRASE_QUESTION_BASED_ON_ITEMS_PROMPT = PromptTemplate(
    template="""You are given a question and a list of items. Rephrase the question to be relevant to the items in the list, if necessary.
    Example 1 (one item):\n
    Question: "Can this be recycled? If not, what is the proper way to dispose of it?"\n
    Items: ["item1"]\n
    Rephrased Question: "Can item1 be recycled? If not, what is the proper way to dispose of it?"\n
    \n

    Example 2 (multiple items):\n
    Question: "Are these items recyclable?"\n
    Items: ["item1", "item2"]\n
    Rephrased Question: "Are item1 and item2 recyclable?"\n
    \n

    Example 3 (question already relevant to items):\n
    Question: "How to recycle thermal flask and plastic bottle?"\n
    Items: ["thermal flask", "plastic bottle"]\n
    Rephrased Question: "How to recycle thermal flask and plastic bottle?"\n
    \n

    Here is the question: {question} \n
    Here is the list of items: {items}
    """,
    input_variables=["question", "items"],
)


rephrase_question_based_on_items_chain = (
    REPHRASE_QUESTION_BASED_ON_ITEMS_PROMPT | LLM | StrOutputParser()
)

INDIV_ITEM_QUESTION_TEMPLATE = """

```is {item} recyclable? If so, please provide the instructions on how to recycle it properly.```

"""


async def rephrase_question_based_on_image_items(state: State, config: RunnableConfig):
    """Rephrase the user's question and update the state with the rephrased question.
    if image is present:
        If user has provided a question, rephrase that question with reference to the items identified from the image, if necessary.
        else if user has not provided a question, create a question using the items identified from the image, based on the template INDIV_ITEM_QUESTION_TEMPLATE.
    else:
        set the state's question to the user's question

    """

    # if user has provided a question, rephrase that question with reference to the items identified from the image.
    # else if user has not provided a question, create a question using the items identified from the image, based on the template INDIV_ITEM_QUESTION_TEMPLATE.
    rephrased_question = (
        await rephrase_question_based_on_items_chain.ainvoke(
            {"question": state.user_question, "items": state.items}
        )
        if state.user_question
        else "\n".join(
            [INDIV_ITEM_QUESTION_TEMPLATE.format(item=item) for item in state.items]
        )
    )
    if config["configurable"].get("debug", False):
        print("\n------- in rephrase_question_based_on_image_items node -------\n")
        print(f"rephrased question based on image items: {rephrased_question}")
    return {"rephrased_question": rephrased_question}


# endregion: rephrase question based on image items

# region: rephrase question based on chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    "If there is no chat history, return the user's question as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

rephrase_question_from_history_chain = contextualize_q_prompt | LLM | StrOutputParser()


async def rephrase_question_based_on_chat_history(state: State, config: RunnableConfig):
    """Rephrase the user's question based on the chat history and update the state with the rephrased question."""

    chat_history = [
        message
        for message in state.messages
        if message.additional_kwargs.get("custom-type") != "image"
    ]

    # use the user's question if the rephrased question is not provided i.e. no image is present. Question will first be rephrased based on image items if image is present, otherwise, there wont be a rephrased question.
    rephrased_question = (
        state.rephrased_question if state.rephrased_question else state.user_question
    )
    rephrased_question = await rephrase_question_from_history_chain.ainvoke(
        {"input": rephrased_question, "chat_history": chat_history}, config
    )
    if config["configurable"].get("debug", False):
        print("\n------- in rephrase_question_based_on_chat_history node -------\n")
        print(f"rephrased question: {rephrased_question}")
    return {"rephrased_question": rephrased_question}


# endregion: rephrase question based on chat history


# region: retrieve docs
async def retrieve_docs(state: State, config: RunnableConfig):
    """Retrieve docs from the database and adds them to the state:
    - based on image items (if image is present)
    - based on question
    """
    retrieved_docs = []
    unique_docs = set()  # to avoid duplicate docs
    # retrieve docs based on image items (if there are image items)
    for item in state.items:
        docs: List[Document] = hybrid_search(item, k=10)
        for doc in docs:
            if doc.page_content not in unique_docs:
                retrieved_docs.append(doc)
                unique_docs.add(doc.page_content)

    # retrieve docs based on user's question
    if rephrased_question := state.rephrased_question:
        docs: List[Document] = hybrid_search(
            rephrased_question, k=10, alpha=1
        )  # perform vector search (alpha = 1) on the question
        for doc in docs:
            if doc.page_content not in unique_docs:
                retrieved_docs.append(doc)
                unique_docs.add(doc.page_content)

    if config["configurable"].get("debug", False):
        print("\n------- in retrieve_docs node-------\n")
        print(f"retrieved docs: {retrieved_docs}")
    return {"retrieved_docs": retrieved_docs}


# endregion: retrieve docs

# region: grade retrieved docs
GRADE_RETRIEVED_DOCS_PROMPT = PromptTemplate(
    template="""You are a grader assessing relevance of retrieved documents to a question. \n
    Return the output as a JSON with an `is_relevant` field being a Boolean True or False to indicate whether the documents contain the answer or not. As a guide, if the `item` field in the retrieved documents approximately contains the items asked in the question, then grade the documents are relevant. \n
    Here are the retrieved documents: \n\n{retrieved_documents} \n\n
    Here is the question: {question} \n
    """,
    input_variables=["question", "retrieved_documents"],
)


class RetrievedDocsRelevance(BaseModel):
    is_relevant: bool = Field(
        description="Boolean True or False to indicate whether the documents contain the answer to answer the question"
    )


grade_retrieved_docs_chain = GRADE_RETRIEVED_DOCS_PROMPT | LLM.with_structured_output(
    schema=RetrievedDocsRelevance
)


async def grade_retrieved_docs(state: State, config: RunnableConfig):
    res = await grade_retrieved_docs_chain.ainvoke(
        {
            "question": state.rephrased_question,
            "retrieved_documents": state.retrieved_docs,
        },
        config=config,
    )
    if config["configurable"].get("debug", False):
        print("\n--------in grade_retrieved_docs node--------\n")
        print(
            f"Is retrieved docs relevant for question {state.rephrased_question}: {res.is_relevant}"
        )
    return {"is_retrieved_docs_relevant": res.is_relevant}


# endregion: grade retrieved docs


# region: perform web search
async def should_do_web_search(
    state: State, config: RunnableConfig
) -> Literal["web_search_needed", "web_search_not_needed"]:
    if config["configurable"].get("debug", False):
        print("\n--------in should_do_web_search node--------")

    if state.is_retrieved_docs_relevant:
        # print("Retrieved docs are relevant, no need to perform web search")
        return "web_search_not_needed"
    else:
        max_web_search_count = config["configurable"].get("max_web_search_count", 4)
        if state.web_search_count < max_web_search_count:
            if config["configurable"].get("debug", False):
                print(
                    f"Current web search count: {state.web_search_count}. Web search needed"
                )
            return "web_search_needed"
        else:
            if config["configurable"].get("debug", False):
                print(
                    f"Max web search count reached: {max_web_search_count}. Stopping web search"
                )
            return "stop_web_search"


async def perform_web_search(state: State, config: RunnableConfig):
    """Perform web search to add documents to the retrieved docs"""

    web_search_docs = await web_search_tool.ainvoke(
        {"query": state.rephrased_question}, config=config
    )
    # print(f"no. of web search docs: {len(web_search_docs)}")
    # print(f"web search docs: {web_search_docs}")

    # convert web search docs to Document objects
    for i in range(len(web_search_docs)):
        web_search_docs[i] = Document(
            page_content=web_search_docs[i]["content"],
            metadata={"source": web_search_docs[i]["url"]},
        )
    retrieved_docs = state.retrieved_docs + web_search_docs

    if config["configurable"].get("debug", False):
        print("\n--------in perform_web_search node--------\n")
        print(f"retrieved_docs: {retrieved_docs}")
    return {
        "retrieved_docs": retrieved_docs,
        "web_search_count": state.web_search_count + 1,
    }


# endregion: perform web search

# region: generate answer
ANS_GEN_PROMPT = PromptTemplate(
    template="""
    You are given question(s) and a context to refer to, to answer the question(s). \n
    For each question, answer the question based on the context. \n
    Answer the question in a clear and concise human-readable format. Include markdown formatting if necessary. \n
    When answering the question, state the source at the bottom of the answer where you got the answer from, from the context. \n
    If the answer is not in the context, do state that you are unable to find the answer from the knowledge base to answer the question, and then use your knowledge to answer the question,  \n
    Example format of answer for each question: \n
    <Answer> 
    \n\n
    Source(s):
    - <Source 1 web url link obtained from `links` field or `source` field in the context. If there is no url link present in the context, return ```Information derived from own knowledge database.``` or ```Information derived from pre-trained knowledge``` if the information was from your own knowledge>
    - ... \n
    \n
    Here are the question(s): \n
    ```
    {questions} \n
    ```
    \n\n

    Here is the context: \n
    ```
    {context}
    ```
    \n\n

    """,
    input_variables=["questions", "context"],
)

ans_gen_chain = ANS_GEN_PROMPT | LLM | StrOutputParser()


async def generate_answer(state: State, config: RunnableConfig):
    """Generate answer from retrieved docs as context. Returns generic answer from LLM if answer is not found in context."""

    answer = await ans_gen_chain.ainvoke(
        {"questions": state.rephrased_question, "context": state.retrieved_docs},
        config=config,
    )
    if config["configurable"].get("debug", False):
        print("\n--------in generate_answer node--------\n")
        print(f"Generated answer: {answer}")
    return {"generated_answer": answer}


# endregion: generate answer

# region: generate answer from LLM
ANS_GEN_FROM_LLM_PROMPT = PromptTemplate(
    template="""
    Use your knowledge to answer the question. \n
    This is how your answer should be formatted: \n
    <Insert your answer here>
    \n\n
    Source:
    - Information obtained from ChatGPT which might be inaccurate. 
    
    \n\n

    Here are the question(s): \n
    ```
    {questions} \n
    ```

    """,
    input_variables=["questions"],
)

ans_gen_from_llm_chain = ANS_GEN_FROM_LLM_PROMPT | LLM | StrOutputParser()


async def generate_answer_from_llm(state: State, config: RunnableConfig):
    """Generate answer from LLM as fallback if answer is not found in context nor web search results."""

    answer = await ans_gen_from_llm_chain.ainvoke(
        {"questions": state.rephrased_question}, config=config
    )
    if config["configurable"].get("debug", False):
        print("\n--------in generate_answer from LLM node--------\n")
        print(f"Generated answer from LLM: {answer}")
    return {"generated_answer": answer}


# endregion: generate answer from LLM


# region: persist chat messages
async def persist_chat_messages(state: State, config: RunnableConfig):
    """Persist chat messages into Firestore"""
    if config["configurable"].get("debug", False):
        print("\n--------in persist_chat_messages node--------\n")

    messages: List[BaseMessage] = []
    # if there is a user image, upload onto cloud storage and persist the image url
    if base64_image := config["configurable"].get("base64_image", None):
        content_type, base64_image_data = extract_image_data(base64_image)
        file_extension = content_type.split("/")[1]
        image_url = upload_base64_image_to_gcs(
            bucket_ref,
            f"{IMAGE_FOLDER}/{uuid.uuid4()}.{file_extension}",
            base64_image_data,
            content_type,
        )
        image_message = HumanMessage(
            content=image_url, additional_kwargs={"custom-type": "image"}
        )
        messages.append(image_message)

    # if user question is present, add to messages
    if user_question := state.user_question:
        messages.append(HumanMessage(content=user_question))

    # if generated answer is present, add to messages
    if generated_answer := state.generated_answer:
        messages.append(AIMessage(content=generated_answer))

    return {
        "messages": messages
    }  # returning list of messages to be added to the existing messages in state automatically


# endregion: persist chat messages
