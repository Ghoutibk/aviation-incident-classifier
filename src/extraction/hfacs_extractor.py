"""Extracteur HFACS basé sur Mistral + LangChain + sortie structurée Pydantic.

Architecture identique au classifier (même patterns), mais :
- Schéma de sortie plus riche (4 niveaux HFACS + facteurs complémentaires)
- Prompt qui guide explicitement le LLM dans la hiérarchie HFACS
- Troncature des rapports longs pour tenir dans le contexte Mistral
"""
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from loguru import logger

from src.extraction.hfacs_schema import ContributingFactors

load_dotenv()

MODEL_NAME = "mistral-small-latest"

# Le prompt système explique HFACS au LLM ET lui donne les règles d'extraction
SYSTEM_PROMPT = """Tu es un expert en analyse d'accidents aéronautiques, spécialiste du modèle HFACS (Human Factors Analysis and Classification System) de Wiegmann & Shappell.

Ta tâche : extraire les facteurs contributifs d'un rapport BEA selon la taxonomie HFACS à 4 niveaux.

LES 4 NIVEAUX HFACS (du plus proche au plus éloigné de l'accident) :

1. UNSAFE ACTS (actes dangereux) : erreurs ou violations COMMISES par le pilote.
   - error : erreur non intentionnelle (ex: mauvaise lecture d'instrument)
   - violation : non-respect délibéré (ex: vol VFR en conditions IMC)

2. PRECONDITIONS (préconditions) : conditions présentes AVANT l'acte dangereux.
   - physical_mental_state : fatigue, stress, maladie, inattention
   - crew_resource_mgmt : communication équipage, coordination
   - environmental : météo, turbulence, luminosité (contexte externe)
   - technological : interface homme-machine, ergonomie

3. UNSAFE SUPERVISION : défaillances dans l'encadrement/formation du pilote.
   Ex : instructeur absent, formation insuffisante, planification inadéquate.

4. ORGANIZATIONAL INFLUENCES : causes au niveau de l'organisation/compagnie.
   Ex : politique de maintenance, pression commerciale, climat de sécurité.

+ Deux listes complémentaires pour aspects non-humains :
- technical_factors : pannes systèmes, défauts matériels (courts, factuels)
- environmental_factors : météo, relief, obstacles (courts, factuels)

RÈGLES :
- N'invente PAS de facteurs : si un niveau n'est pas évoqué dans le rapport, laisse la liste vide.
- Sois FACTUEL : cite ce qui est explicitement mentionné ou fortement suggéré.
- Chaque description doit être COURTE (1-2 phrases max).
- Le champ primary_cause doit faire MAX 200 caractères.
- Si tu hésites, baisse le confidence."""

USER_PROMPT = """RAPPORT D'INCIDENT BEA

Référence : {bea_reference}
Immatriculation : {aircraft_registration}

{report_text}

Extrait les facteurs contributifs selon la taxonomie HFACS."""


def build_extractor():
    """Construit la chaîne d'extraction LangChain.

    with_structured_output() force Mistral à retourner un JSON conforme
    au schéma Pydantic. Si le JSON est invalide, LangChain retry.
    """
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY n'est pas défini dans .env")

    llm = ChatMistralAI(model=MODEL_NAME, temperature=0.1)
    # .with_structured_output lie le schéma au LLM : la sortie sera parsée en ContributingFactors
    structured_llm = llm.with_structured_output(ContributingFactors)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])
    # Le pipe | crée la chaîne : prompt → llm → objet Pydantic
    return prompt | structured_llm


def extract_factors(
    report_text: str,
    bea_reference: str | None = None,
    aircraft_registration: str | None = None,
    max_chars: int = 15000,
) -> ContributingFactors:
    """Extrait les facteurs HFACS d'un rapport.

    On tronque les rapports longs à 15k caractères pour rester sous
    le contexte Mistral Small (32k tokens) avec marge de sécurité.
    Pour des rapports plus longs, une stratégie map-reduce serait
    nécessaire (découpage par section + fusion), mais c'est hors scope POC.
    """
    if len(report_text) > max_chars:
        logger.warning(f"Rapport tronqué : {len(report_text)} → {max_chars} caractères")
        report_text = report_text[:max_chars] + "\n\n[...rapport tronqué...]"

    chain = build_extractor()
    
    result = chain.invoke({
        "report_text": report_text,
        "bea_reference": bea_reference or "N/A",
        "aircraft_registration": aircraft_registration or "N/A",
    })
    return result