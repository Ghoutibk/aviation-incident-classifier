"""Retriever : recherche sémantique dans ChromaDB."""
from dataclasses import dataclass

from loguru import logger

from src.vector_store.chroma_client import (
    get_chroma_client,
    get_embeddings,
    get_or_create_collection,
)


@dataclass
class RetrievedChunk:
    text: str
    report_filename: str
    bea_reference: str
    chunk_index: int
    similarity: float 


_embedder = None
_collection = None


def _get_components():
    global _embedder, _collection
    if _embedder is None:
        _embedder = get_embeddings()
    if _collection is None:
        client = get_chroma_client()
        _collection = get_or_create_collection(client)
    return _embedder, _collection


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    embedder, collection = _get_components()

    query_embedding = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append(RetrievedChunk(
            text=results["documents"][0][i],
            report_filename=results["metadatas"][0][i]["report_filename"],
            bea_reference=results["metadatas"][0][i]["bea_reference"],
            chunk_index=results["metadatas"][0][i]["chunk_index"],
            similarity=1 - results["distances"][0][i],
        ))

    logger.debug(f"Retrieved {len(chunks)} chunks pour : '{query[:60]}...'")
    return chunks