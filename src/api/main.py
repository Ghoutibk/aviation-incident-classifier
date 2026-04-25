"""API REST FastAPI : expose les fonctions du projet comme endpoints HTTP.

Architecture :
- POST /classify : classifie un rapport (texte libre)
- POST /extract : extrait les facteurs HFACS d'un rapport
- POST /ask : pose une question au corpus (RAG)
- GET /reports : liste les rapports en base avec pagination
- GET /reports/{id}/classification : détail de classification d'un rapport
- GET /weak-signals : retourne les clusters de signaux faibles
- GET /regulatory-alerts : retourne les alertes réglementaires

Pourquoi FastAPI et pas Flask ?
- Validation automatique des entrées/sorties via Pydantic (cohérent avec notre stack)
- Documentation interactive auto-générée sur /docs (Swagger UI)
- Typage Python natif, plus propre
- Performant (async si besoin)
"""
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select

from src.classification.classifier import classify_report
from src.classification.schemas import IncidentClassification
from src.db.models import Classification, FactorsExtraction, RegulatoryAlert, Report, engine
from src.extraction.hfacs_extractor import extract_factors
from src.extraction.hfacs_schema import ContributingFactors
from src.rag.chain import ask as rag_ask
from src.weak_signals.clustering import detect_weak_signals

app = FastAPI(
    title="Aviation Incident Classifier API",
    description="API pour classification et analyse de rapports BEA via LLM",
    version="1.0.0",
)

# CORS : autorise Streamlit (et tout autre front) à appeler l'API depuis un navigateur
# En prod, on restreint à un domaine précis, mais ici on ouvre pour simplicité
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Modèles de requête (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    """Corps de requête pour POST /classify."""
    report_text: str
    bea_reference: str | None = None
    aircraft_registration: str | None = None


class ExtractRequest(BaseModel):
    """Corps de requête pour POST /extract."""
    report_text: str
    bea_reference: str | None = None
    aircraft_registration: str | None = None


class AskRequest(BaseModel):
    """Corps de requête pour POST /ask (RAG)."""
    question: str
    k: int = 5  # nombre de chunks à récupérer


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Endpoint racine : retourne un message de bienvenue."""
    return {
        "service": "Aviation Incident Classifier API",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    """Healthcheck : utile pour un monitoring / k8s readiness probe."""
    return {"status": "ok"}


@app.post("/classify", response_model=IncidentClassification)
def classify(request: ClassifyRequest):
    """Classifie un rapport selon les domaines de risque et la criticité."""
    try:
        result = classify_report(
            report_text=request.report_text,
            bea_reference=request.bea_reference,
            aircraft_registration=request.aircraft_registration,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract", response_model=ContributingFactors)
def extract(request: ExtractRequest):
    """Extrait les facteurs contributifs HFACS d'un rapport."""
    try:
        result = extract_factors(
            report_text=request.report_text,
            bea_reference=request.bea_reference,
            aircraft_registration=request.aircraft_registration,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_corpus(request: AskRequest):
    """Pose une question au corpus BEA via le pipeline RAG."""
    try:
        response = rag_ask(request.question, k=request.k)
        return {
            "answer": response.answer,
            "sources": [
                {
                    "bea_reference": s.bea_reference,
                    "filename": s.report_filename,
                    "similarity": s.similarity,
                    "excerpt": s.text[:300],
                }
                for s in response.sources
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports")
def list_reports(
    limit: int = Query(20, le=200),
    offset: int = 0,
):
    """Liste paginée des rapports en base."""
    with Session(engine) as session:
        reports = session.exec(
            select(Report).offset(offset).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "bea_reference": r.bea_reference,
                "aircraft_registration": r.aircraft_registration,
                "page_count": r.page_count,
            }
            for r in reports
        ]


@app.get("/reports/{report_id}")
def get_report(report_id: int):
    """Détail d'un rapport avec sa classification et son extraction."""
    with Session(engine) as session:
        report = session.get(Report, report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Rapport introuvable")

        classification = session.exec(
            select(Classification).where(Classification.report_id == report_id)
        ).first()

        extraction = session.exec(
            select(FactorsExtraction).where(FactorsExtraction.report_id == report_id)
        ).first()

        return {
            "report": {
                "id": report.id,
                "filename": report.filename,
                "bea_reference": report.bea_reference,
                "page_count": report.page_count,
            },
            "classification": {
                "domains": json.loads(classification.domains) if classification else None,
                "criticality": classification.criticality if classification else None,
                "confidence": classification.confidence if classification else None,
                "reasoning": classification.reasoning if classification else None,
            } if classification else None,
            "extraction": {
                "primary_cause": extraction.primary_cause if extraction else None,
                "unsafe_acts": json.loads(extraction.unsafe_acts) if extraction else [],
                "preconditions": json.loads(extraction.preconditions) if extraction else [],
            } if extraction else None,
        }


@app.get("/weak-signals")
def weak_signals():
    """Retourne les clusters de signaux faibles (rapports récents émergents)."""
    try:
        signals = detect_weak_signals()
        return [
            {
                "cluster_id": s.cluster_id,
                "size": s.size,
                "recent_ratio": s.recent_ratio,
                "report_refs": s.report_refs[:10],
            }
            for s in signals
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regulatory-alerts")
def regulatory_alerts(limit: int = Query(20, le=100)):
    """Liste des alertes réglementaires EASA/DGAC en base."""
    with Session(engine) as session:
        alerts = session.exec(
            select(RegulatoryAlert).limit(limit)
        ).all()
        return [
            {
                "id": a.id,
                "source": a.source,
                "title": a.title,
                "summary": a.summary,
                "themes": json.loads(a.relevance_themes),
                "url": a.url,
            }
            for a in alerts
        ]