"""Compare les annotations manuelles aux prédictions LLM et calcule les métriques."""
import csv
import json
from pathlib import Path

from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sqlmodel import Session, select

from src.classification.taxonomy import Criticality, RiskDomain
from src.db.models import Classification, engine

ANNOTATION_PATH = Path("data/annotated/annotation_set.csv")
RESULTS_PATH = Path("data/annotated/evaluation_results.json")

ALL_DOMAINS = [d.value for d in RiskDomain]
ALL_CRITICALITIES = [c.value for c in Criticality]


def load_annotations() -> list[dict]:
    """Charge les annotations manuelles depuis le CSV."""
    annotations = []
    with open(ANNOTATION_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            # Nettoyer les espaces autour des clés
            row = {k.strip(): v for k, v in row.items()}
            # Skip les lignes non annotées
            if not row.get("true_domains") or not row.get("true_criticality"):
                logger.warning(f"Ligne non annotée ignorée : {row.get('filename', 'unknown')}")
                continue
            annotations.append({
                "filename": row["filename"].strip(),
                "true_domains": set(d.strip() for d in row["true_domains"].split("|")),
                "true_criticality": row["true_criticality"].strip(),
                "difficulty": row.get("difficulty", "").strip(),
                "notes": row.get("notes", "").strip(),
            })
    logger.info(f"{len(annotations)} annotations chargées")
    return annotations


def load_predictions(filenames: list[str]) -> dict[str, dict]:
    """Charge les prédictions LLM pour les rapports annotés."""
    predictions = {}
    with Session(engine) as session:
        for filename in filenames:
            stmt = select(Classification).where(
                Classification.report_filename == filename
            )
            pred = session.exec(stmt).first()
            if pred:
                predictions[filename] = {
                    "domains": set(json.loads(pred.domains)),
                    "criticality": pred.criticality,
                    "confidence": pred.confidence,
                    "reasoning": pred.reasoning,
                }
    logger.info(f"{len(predictions)} prédictions LLM récupérées")
    return predictions


def evaluate_criticality(annotations: list[dict], predictions: dict) -> dict:
    """Évalue la classification de criticité (mono-label)."""
    y_true, y_pred = [], []
    for ann in annotations:
        if ann["filename"] not in predictions:
            continue
        y_true.append(ann["true_criticality"])
        y_pred.append(predictions[ann["filename"]]["criticality"])

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=ALL_CRITICALITIES, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=ALL_CRITICALITIES, zero_division=0)

    logger.info("")
    logger.info("=" * 60)
    logger.info("CRITICALITÉ (mono-label)")
    logger.info("=" * 60)
    logger.info(f"Accuracy : {accuracy:.3f}")
    logger.info(f"F1 macro : {f1_macro:.3f}")
    logger.info(f"F1 weighted : {f1_weighted:.3f}")
    logger.info("")
    logger.info("Rapport détaillé par classe :")
    logger.info("\n" + classification_report(
        y_true, y_pred, labels=ALL_CRITICALITIES, zero_division=0
    ))
    logger.info("Matrice de confusion (lignes=vérité, colonnes=prédictions) :")
    logger.info(f"Ordre des classes : {ALL_CRITICALITIES}")
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CRITICALITIES)
    logger.info(f"\n{cm}")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "classes": ALL_CRITICALITIES,
    }


def evaluate_domains(annotations: list[dict], predictions: dict) -> dict:
    """Évalue la classification des domaines (multi-label)."""
    y_true_sets, y_pred_sets = [], []
    for ann in annotations:
        if ann["filename"] not in predictions:
            continue
        y_true_sets.append(ann["true_domains"])
        y_pred_sets.append(predictions[ann["filename"]]["domains"])

    # Binarisation : transforme les sets en vecteurs 0/1
    mlb = MultiLabelBinarizer(classes=ALL_DOMAINS)
    y_true_bin = mlb.fit_transform(y_true_sets)
    y_pred_bin = mlb.transform(y_pred_sets)

    # F1 macro : moyenne non pondérée des F1 de chaque domaine
    f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    # Accuracy subset : tout juste pour un rapport = 1, sinon 0 (très strict)
    accuracy_subset = accuracy_score(y_true_bin, y_pred_bin)

    # Précision/rappel/F1 par domaine
    per_class = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average=None, labels=range(len(ALL_DOMAINS)), zero_division=0
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("DOMAINES (multi-label)")
    logger.info("=" * 60)
    logger.info(f"F1 macro : {f1_macro:.3f}")
    logger.info(f"F1 micro : {f1_micro:.3f}")
    logger.info(f"Accuracy (subset exact) : {accuracy_subset:.3f}")
    logger.info("")
    logger.info("Détail par domaine :")
    logger.info(f"{'Domaine':<20} {'Précision':>10} {'Rappel':>10} {'F1':>10} {'Support':>10}")
    for i, domain in enumerate(ALL_DOMAINS):
        prec, rec, f1, support = (
            per_class[0][i], per_class[1][i], per_class[2][i], per_class[3][i]
        )
        logger.info(f"{domain:<20} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10}")

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "accuracy_subset": accuracy_subset,
        "per_class": {
            ALL_DOMAINS[i]: {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i in range(len(ALL_DOMAINS))
        },
    }


def show_errors(annotations: list[dict], predictions: dict) -> list[dict]:
    """Affiche les cas où le LLM a divergé de la vérité terrain."""
    errors = []
    for ann in annotations:
        if ann["filename"] not in predictions:
            continue
        pred = predictions[ann["filename"]]

        domain_match = ann["true_domains"] == pred["domains"]
        crit_match = ann["true_criticality"] == pred["criticality"]

        if not domain_match or not crit_match:
            errors.append({
                "filename": ann["filename"],
                "true_domains": sorted(ann["true_domains"]),
                "pred_domains": sorted(pred["domains"]),
                "true_criticality": ann["true_criticality"],
                "pred_criticality": pred["criticality"],
                "difficulty": ann["difficulty"],
                "reasoning": pred["reasoning"][:200],
            })

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"ERREURS DU LLM ({len(errors)}/{len(annotations)} cas)")
    logger.info("=" * 60)
    for err in errors:
        logger.info("")
        logger.info(f"📄 {err['filename']} (difficulté : {err['difficulty']})")
        logger.info(f"  VÉRITÉ  : domaines={err['true_domains']} | criticité={err['true_criticality']}")
        logger.info(f"  PRÉDIT  : domaines={err['pred_domains']} | criticité={err['pred_criticality']}")
        logger.info(f"  Raison LLM : {err['reasoning']}...")

    return errors


def main() -> None:
    if not ANNOTATION_PATH.exists():
        logger.error(f"Fichier d'annotation introuvable : {ANNOTATION_PATH}")
        return

    # Charger annotations + prédictions
    annotations = load_annotations()
    filenames = [ann["filename"] for ann in annotations]
    predictions = load_predictions(filenames)

    # Filtrer sur les rapports qui ont les deux
    common = [ann for ann in annotations if ann["filename"] in predictions]
    if len(common) < len(annotations):
        logger.warning(
            f"{len(annotations) - len(common)} rapports annotés sans prédiction LLM"
        )

    results = {
        "n_samples": len(common),
        "criticality": evaluate_criticality(annotations, predictions),
        "domains": evaluate_domains(annotations, predictions),
        "errors": show_errors(annotations, predictions),
    }

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.success(f"Résultats sauvegardés dans {RESULTS_PATH}")


if __name__ == "__main__":
    main()