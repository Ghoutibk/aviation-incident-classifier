"""Taxonomie de classification des rapports d'incidents BEA."""
from enum import Enum


class RiskDomain(str, Enum):

    HUMAN_FACTOR = "human_factor"          # pilotage, decision, fatigue, CRM
    TECHNICAL = "technical"                # panne systeme, avionique, structure
    MAINTENANCE = "maintenance"            # entretien, erreur de montage, verif
    WEATHER = "weather"                    # meteo, givrage, turbulences, vent
    OPERATIONS = "operations"              # ATC, navigation, procédures, piste
    INFRASTRUCTURE = "infrastructure"      # aerodrome, obstacles, ligne electrique


class Criticality(str, Enum):
    MINOR = "minor"                        # incident, pas de blesse, dommages legers
    MAJOR = "major"                        # incident grave, blesses, dommages importants
    CATASTROPHIC = "catastrophic"          # accident avec deces ou destruction totale


DOMAIN_DESCRIPTIONS = {
    RiskDomain.HUMAN_FACTOR: "Facteurs humains : erreurs de pilotage, prise de décision, fatigue, communication, formation, CRM (Crew Resource Management).",
    RiskDomain.TECHNICAL: "Défaillances techniques de l'aéronef : panne moteur, système avionique, structure, commandes de vol (hors maintenance).",
    RiskDomain.MAINTENANCE: "Problèmes liés à la maintenance : erreur d'assemblage, pièce défectueuse non détectée, absence de double vérification, non-conformité aux procédures d'entretien.",
    RiskDomain.WEATHER: "Conditions météorologiques : givrage, turbulences, vent fort, cisaillement, conditions IMC non prévues, orage.",
    RiskDomain.OPERATIONS: "Opérations aériennes : contrôle aérien (ATC), navigation, non-respect des procédures, incursion sur piste, perte d'espacement.",
    RiskDomain.INFRASTRUCTURE: "Infrastructure et environnement au sol : état de la piste, obstacles, balisage, lignes électriques, aérodrome non éclairé.",
}

CRITICALITY_DESCRIPTIONS = {
    Criticality.MINOR: "Incident mineur : aucun blessé, dommages matériels limités, poursuite du vol ou atterrissage normal.",
    Criticality.MAJOR: "Incident majeur ou grave : blessés (non mortels), dommages importants à l'aéronef, atterrissage forcé, évacuation d'urgence.",
    Criticality.CATASTROPHIC: "Accident catastrophique : décès, destruction totale ou quasi-totale de l'aéronef.",
}