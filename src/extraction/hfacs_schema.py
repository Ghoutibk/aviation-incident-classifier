"""Schéma Pydantic pour l'extraction HFACS des facteurs contributifs.

HFACS = Human Factors Analysis and Classification System.
C'est un modèle standard en safety aéronautique qui décompose les accidents
en 4 niveaux hiérarchiques (du plus proche de l'accident au plus éloigné)."""

from pydantic import BaseModel, Field


class UnsafeAct(BaseModel):
    """Niveau 1 HFACS : actes dangereux commis par l'opérateur (pilote).

    Exemples dans un rapport BEA :
    - "Le pilote a omis de vérifier la quantité de carburant" (erreur)
    - "Le pilote a volontairement survolé une zone interdite" (violation)
    """
    description: str = Field(
        description="Description courte et factuelle de l'acte dangereux (1-2 phrases)."
    )
    category: str = Field(
        description="Catégorie HFACS : 'error' (erreur non intentionnelle) ou 'violation' (non-respect délibéré)."
    )


class Precondition(BaseModel):
    """Niveau 2 HFACS : conditions qui ont précédé l'acte dangereux.

    Peut inclure l'état physique/mental du pilote, l'environnement,
    la coordination d'équipage, les conditions techniques.
    """
    description: str = Field(
        description="Description de la précondition (fatigue, stress, météo défavorable, etc.)."
    )
    category: str = Field(
        description="Catégorie : 'physical_mental_state', 'crew_resource_mgmt', 'environmental', 'technological'."
    )


class UnsafeSupervision(BaseModel):
    """Niveau 3 HFACS : défaillances dans la supervision/encadrement.

    Exemples :
    - Formation insuffisante du pilote
    - Planification du vol inadéquate
    - Absence de contrôle par un instructeur
    """
    description: str = Field(
        description="Description de la défaillance de supervision."
    )


class OrganizationalInfluence(BaseModel):
    """Niveau 4 HFACS : influences organisationnelles (niveau le plus haut).

    Exemples :
    - Politique de maintenance insuffisante de l'exploitant
    - Pression commerciale poussant à voler dans de mauvaises conditions
    - Climat de sécurité défaillant dans l'organisation
    """
    description: str = Field(
        description="Description de l'influence organisationnelle."
    )


class ContributingFactors(BaseModel):
    """Sortie complète de l'extraction HFACS pour un rapport.

    Les 4 listes peuvent être vides si aucun facteur n'est identifié
    à ce niveau — c'est fréquent en aviation générale (peu d'organisation).

    Les champs `technical_factors` et `environmental_factors` s'ajoutent
    à HFACS classique pour capturer les aspects non-humains fréquents
    dans les rapports BEA (panne matérielle, météo, etc.).
    """
    # Les 4 niveaux HFACS (toujours présents, même si vides)
    unsafe_acts: list[UnsafeAct] = Field(
        default_factory=list,
        description="Niveau 1 : actes dangereux du pilote (erreurs ou violations).",
    )
    preconditions: list[Precondition] = Field(
        default_factory=list,
        description="Niveau 2 : préconditions (état du pilote, environnement, équipage).",
    )
    unsafe_supervision: list[UnsafeSupervision] = Field(
        default_factory=list,
        description="Niveau 3 : défaillances de supervision/encadrement.",
    )
    organizational_influences: list[OrganizationalInfluence] = Field(
        default_factory=list,
        description="Niveau 4 : influences organisationnelles.",
    )

    # Champs complémentaires pour aspects non-humains (fréquents en aéro)
    technical_factors: list[str] = Field(
        default_factory=list,
        description="Facteurs techniques non-liés à l'humain (panne système, usure pièce).",
    )
    environmental_factors: list[str] = Field(
        default_factory=list,
        description="Facteurs environnementaux (météo, relief, obstacle).",
    )

    # Synthèse
    primary_cause: str = Field(
        description="Cause principale identifiée par le rapport, en 1 phrase courte (< 200 caractères).",
        max_length=200,
    )
    confidence: float = Field(
        description="Niveau de confiance dans l'extraction, entre 0 et 1.",
        ge=0.0,
        le=1.0,
    )