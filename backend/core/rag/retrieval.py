import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_cohere import CohereEmbeddings

from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBED_MODEL

_embeddings = None
_vector_store = None
_retriever = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="embed-english-v3.0"
        )
    return _embeddings

def get_retriever():
    global _retriever, _vector_store
    if _retriever is not None:
        return _retriever

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
    )

    embeddings = get_embeddings()

    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    _retriever = _vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4},
    )
    return _retriever
