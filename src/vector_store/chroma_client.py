"""Client ChromaDB : wrapper autour de la vector DB"""
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

CHROMA_PATH = Path("data/chroma")

COLLECTION_NAME = "bea_reports"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Collection '{COLLECTION_NAME}' : {collection.count()} vecteurs")
    return collection