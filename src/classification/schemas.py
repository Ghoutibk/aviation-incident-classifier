"""Schémas Pydantic pour la sortie structurée du classifier."""
from pydantic import BaseModel, Field
from src.classification.taxonomy import Criticality, RiskDomain


class IncidentClassification(BaseModel):

    domains: list[RiskDomain] = Field(
        description="Liste des domaines de risque identifiés dans le rapport. Multi-label : un incident peut impliquer plusieurs domaines (ex: maintenance + facteur humain).",
        min_length=1,
    )
    criticality: Criticality = Field(
        description="Niveau de criticité global de l'incident, basé sur les conséquences (blessés, dommages, décès)."
    )
    confidence: float = Field(
        description="Niveau de confiance dans la classification, entre 0 et 1.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Justification brève (2-3 phrases) expliquant pourquoi ces domaines et cette criticité ont été retenus.",
        max_length=500,
    )