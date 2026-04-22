"""Classifier de rapports d'incidents basé sur Mistral via LangChain."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from loguru import logger

from src.classification.schemas import IncidentClassification
from src.classification.taxonomy import (
    CRITICALITY_DESCRIPTIONS,
    DOMAIN_DESCRIPTIONS,
)

load_dotenv()

MODEL_NAME = "mistral-small-latest"

#Génère la description de la taxonomie pour le prompt
def _build_taxonomy_block() -> str:
    domains_txt = "\n".join(
        f"- {domain.value}: {desc}"
        for domain, desc in DOMAIN_DESCRIPTIONS.items()
    )
    criticality_txt = "\n".join(
        f"- {crit.value}: {desc}"
        for crit, desc in CRITICALITY_DESCRIPTIONS.items()
    )
    return f"DOMAINES DE RISQUE :\n{domains_txt}\n\nNIVEAUX DE CRITICITÉ :\n{criticality_txt}"


SYSTEM_PROMPT = """Tu es un expert en sécurité aérienne qui analyse des rapports d'incidents du BEA (Bureau d'Enquêtes et d'Analyses).

Ta tâche : classer chaque rapport selon deux dimensions :
1. Les DOMAINES DE RISQUE impliqués (multi-label : un incident peut en avoir plusieurs)
2. Le niveau de CRITICITÉ global (une seule valeur)

{taxonomy}

RÈGLES :
- Lis attentivement le rapport et identifie TOUS les domaines de risque pertinents (pas seulement le principal).
- Évalue la criticité selon les CONSÉQUENCES réelles (blessés, décès, dommages), pas le potentiel.
- Sois précis et factuel dans le reasoning, cite des éléments concrets du rapport.
- Si tu hésites, baisse le niveau de confidence."""


USER_PROMPT = """Voici un rapport d'incident BEA à classifier.

Référence : {bea_reference}
Immatriculation : {aircraft_registration}

CONTENU DU RAPPORT :
{report_text}

Classifie ce rapport selon la taxonomie définie."""

#Construit la chaîne de classification LangChain
def build_classifier():
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY n'est pas defini dans .env")

    llm = ChatMistralAI(model=MODEL_NAME, temperature=0.1)
    structured_llm = llm.with_structured_output(IncidentClassification)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]).partial(taxonomy=_build_taxonomy_block())

    return prompt | structured_llm

#Classifie un rapport. Tronque si trop long pour tenir dans le contexte
def classify_report(
    report_text: str,
    bea_reference: str | None = None,
    aircraft_registration: str | None = None,
    max_chars: int = 15000,
) -> IncidentClassification:
                                    
    if len(report_text) > max_chars:
        logger.warning(
            f"Rapport tronqué : {len(report_text)} → {max_chars} caractères"
        )
        report_text = report_text[:max_chars] + "\n\n[...rapport tronqué...]"

    chain = build_classifier()
    result = chain.invoke({
        "report_text": report_text,
        "bea_reference": bea_reference or "N/A",
        "aircraft_registration": aircraft_registration or "N/A",
    })
    return result