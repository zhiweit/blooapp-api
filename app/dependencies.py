import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from google.cloud import firestore, storage
from app.models import ItemNames, Item
from langchain_core.documents import Document
import json
import typesense
from google.cloud.storage import Bucket
import base64
import io
import re

load_dotenv()

BUCKET_NAME = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")
IMAGE_FOLDER = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET_IMAGE_FOLDER")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
TYPESENSE_COLLECTION = os.getenv("TYPESENSE_COLLECTION_NAME")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if BUCKET_NAME is None:
    raise ValueError("BUCKET_NAME is not set")
if IMAGE_FOLDER is None:
    raise ValueError("IMAGE_FOLDER is not set")
if TYPESENSE_API_KEY is None:
    raise ValueError("TYPESENSE_API_KEY is not set")
if TYPESENSE_COLLECTION is None:
    raise ValueError("TYPESENSE_COLLECTION is not set")
if TYPESENSE_HOST is None:
    raise ValueError("TYPESENSE_HOST is not set")
if TAVILY_API_KEY is None:
    raise ValueError("TAVILY_API_KEY is not set")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
client = firestore.AsyncClient()
ITEMS_COLLECTION = client.collection("wasteType")

vision_model = ChatOpenAI(model="gpt-4o", temperature=0)
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
qa_model_item_output = LLM.with_structured_output(Item, method="json_mode")

# region: typesense
node = {
    "host": TYPESENSE_HOST,  # For Typesense Cloud use xxx.a1.typesense.net
    "port": "443",  # For Typesense Cloud use 443
    "protocol": "https",  # For Typesense Cloud use https
}
typesense_client = typesense.Client(
    {"nodes": [node], "api_key": TYPESENSE_API_KEY, "connection_timeout_seconds": 5}
)


def hybrid_search(query: str, k: int = 10, alpha: float = 0.3):
    """Perform hybrid search on the Typesense vector database

    Keyword arguments:
    query: str -- the query to search for
    k: int -- the number of results to return
    alpha: float [0,1] -- the weight given to the semantic (vector) search. (1 - alpha) is the weight given to the keyword search.
    """

    query_obj = {
        "collection": TYPESENSE_COLLECTION,
        "q": query,
        "query_by": "vec,item,instructions",
        "prefix": "false",
        "vector_query": f"vec:([], alpha: {alpha})",  # alpha is weight given to semantic (vector) search, (1 - alpha) is weight given to keyword search
        "exclude_fields": "vec",
        "limit": k,
    }

    common_search_params = {}

    response = typesense_client.multi_search.perform(
        {"searches": [query_obj]}, common_search_params
    )

    docs = []
    for hit in response["results"][0]["hits"]:
        document = hit["document"]
        content = json.dumps(
            {
                "item": document["item"],
                "instructions": document["instructions"],
                "material": document["material"],
                "recyclable": document["recyclable"],
                "links": document["links"],
            }
        )
        # score = hit['hybrid_search_info']['rank_fusion_score']
        # docs.append((Document(page_content=content), score))
        docs.append(Document(page_content=content, metadata={"source": "database"}))
    return docs


# endregion: typesense

# region: tavily
web_search_tool = TavilySearchResults(
    include_domains=["https://recyclopedia.sg/items"],
    search_depth="advanced",
    max_results=10,
    include_answer=True,
)
# endregion: tavily

# region: cloud storage
storage_client = storage.Client()
bucket_ref = storage_client.bucket(BUCKET_NAME)


def extract_image_data(data_url):
    # Regular expression pattern to match the content type and base64 data
    pattern = r"data:(?P<content_type>[\w/]+);base64,(?P<base64_data>.*)"

    # Try to match the pattern
    match = re.match(pattern, data_url)

    if match:
        # Extract content type and base64 data
        content_type = match.group("content_type")
        base64_data = match.group("base64_data")
        return content_type, base64_data
    else:
        raise ValueError(
            f"Invalid data URL format for base64 image. Example expected format: data:<content-type>/<image-format>;base64,<base64-image-data> \n Got: {data_url[:30]}..."
        )


def upload_base64_image_to_gcs(
    bucket_ref: Bucket,
    destination_blob_name: str,
    base64_image_data: str,
    content_type: str = "image/jpeg",
):
    """Uploads a base64 encoded image to Google Cloud Storage.

    Args:
        bucket_ref (Bucket): The bucket reference to upload the image to.
        destination_blob_name (str): The name of the blob to create in GCS.
        base64_image_data (str): The base64 encoded image string.
        content_type (str): The content type of the image. Defaults to "image/jpeg".
    """

    # Decode the base64 string
    img_data = base64.b64decode(base64_image_data)

    # Create a file-like object from the decoded data
    file_obj = io.BytesIO(img_data)

    # Create a new blob and upload the file's content
    blob = bucket_ref.blob(destination_blob_name)
    blob.upload_from_file(file_obj, content_type=content_type)

    # Make the blob publicly accessible
    blob.make_public()

    # Get the public URL
    public_url = blob.public_url

    # print(f"Public URL: {public_url}")
    return public_url


# endregion: cloud storage
