from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings

from app.models.image_response import ImageResponse

load_dotenv()
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

vision_model = ChatOpenAI(model="gpt-4o", temperature=0)

index_name = os.getenv("PINECONE_INDEX_NAME")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embedder
)

# max marginal search to encourage document diversity when retrieving (but slows down retrieval process)
# retriever = docsearch.as_retriever(
#     search_type="mmr", search_kwargs={"k": 6, "fetch_k": 10}
# )
retriever = docsearch.as_retriever()

qa_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_model_json_output = qa_model.with_structured_output(
    ImageResponse, method="json_mode"
)
