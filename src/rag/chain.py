"""Pipeline RAG complet : question → retrieval → génération avec citations.

C'est la brique 'A + G' de RAG :
- A (Augmented) : on enrichit le prompt LLM avec les chunks retrouvés
- G (Generation) : Mistral répond en s'appuyant uniquement sur ces chunks
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from loguru import logger

from src.rag.retriever import RetrievedChunk, retrieve

load_dotenv()

MODEL_NAME = "mistral-small-latest"

# Prompt RAG clean :
# - Le LLM doit s'appuyer SUR les sources (pas inventer)
# - Doit citer explicitement avec [BEA-XXXX]
# - Dire "je ne sais pas" si la réponse n'est pas dans les sources
RAG_PROMPT = """Tu es un assistant spécialisé en sécurité aérienne, qui répond à des questions en t'appuyant EXCLUSIVEMENT sur des extraits de rapports du BEA (Bureau d'Enquêtes et d'Analyses).

QUESTION :
{question}

EXTRAITS PERTINENTS :
{context}

RÈGLES STRICTES :
1. Tu réponds UNIQUEMENT avec des informations présentes dans les extraits ci-dessus.
2. Si la réponse n'est pas dans les extraits, réponds : "Je ne trouve pas cette information dans le corpus disponible."
3. Cite tes sources entre crochets en utilisant la référence BEA (ex: [BEA2021-0088]).
4. Réponds en français, de manière concise et factuelle (2-5 phrases).
5. N'invente JAMAIS de chiffres, dates ou conclusions non présents dans les extraits."""


@dataclass
class RAGResponse:
    """Réponse complète d'un appel RAG."""
    answer: str
    sources: list[RetrievedChunk]  # pour afficher les sources dans l'UI


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Formate les chunks récupérés en un bloc de contexte pour le prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"--- Extrait {i} [{chunk.bea_reference}] (similarité {chunk.similarity:.2f}) ---\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def ask(question: str, k: int = 5) -> RAGResponse:
    """Pose une question au corpus BEA, obtient une réponse sourcée.

    Pipeline complet :
    1. Retrieval des k chunks les plus pertinents
    2. Construction du prompt avec les extraits
    3. Appel Mistral qui génère la réponse en citant
    """
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY n'est pas défini dans .env")

    # 1. Retrieval
    chunks = retrieve(question, k=k)

    if not chunks:
        return RAGResponse(
            answer="Aucun extrait trouvé dans le corpus pour cette question.",
            sources=[],
        )

    # 2. Formater le contexte
    context = format_context(chunks)

    # 3. Construction du prompt + appel LLM
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatMistralAI(model=MODEL_NAME, temperature=0.2)
    chain = prompt | llm

    response = chain.invoke({"question": question, "context": context})
    answer = response.content

    logger.info(f"RAG answer ({len(chunks)} sources): {answer[:100]}...")

    return RAGResponse(answer=answer, sources=chunks)