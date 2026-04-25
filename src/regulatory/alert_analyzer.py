"""Analyse LLM des alertes réglementaires : résumé + classification thématique."""
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = "mistral-small-latest"

# Thèmes qu'on cherche à matcher dans les alertes
THEMES = [
    "moteur",
    "carburant",
    "givrage",
    "avionique",
    "structure",
    "facteurs_humains",
    "maintenance",
    "ATC_operations",
    "conditions_meteo",
    "formation_pilotage",
]


class AlertAnalysis(BaseModel):
    """Analyse d'une alerte réglementaire."""
    summary: str = Field(
        description="Résumé en 2-3 phrases (max 400 caractères).",
        max_length=400,
    )
    themes: list[str] = Field(
        description=f"Thèmes correspondants parmi : {THEMES}",
        default_factory=list,
    )
    relevance_score: float = Field(
        description="Pertinence globale (0-1) pour un exploitant aérien.",
        ge=0.0,
        le=1.0,
    )


SYSTEM_PROMPT = f"""Tu es un analyste en sécurité aérienne qui synthétise des alertes réglementaires européennes.

THÈMES DISPONIBLES : {', '.join(THEMES)}

Pour chaque alerte :
1. Rédige un résumé FACTUEL et COURT (2-3 phrases, max 400 caractères).
2. Identifie les thèmes pertinents dans la liste ci-dessus (peut être vide).
3. Évalue la pertinence globale pour un exploitant (0 = anecdotique, 1 = critique)."""


def analyze_alert(title: str, content: str) -> AlertAnalysis:
    """Appelle Mistral pour analyser une alerte."""
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY manquante")

    llm = ChatMistralAI(model=MODEL_NAME, temperature=0.1)
    structured = llm.with_structured_output(AlertAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "TITRE: {title}\n\nCONTENU:\n{content}\n\nAnalyse cette alerte."),
    ])
    return (prompt | structured).invoke({"title": title, "content": content[:3000]})