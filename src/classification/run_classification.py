"""Classifie tous les rapports de la base avec Mistral."""
import json
import time

from loguru import logger
from sqlmodel import Session, select

from src.classification.classifier import MODEL_NAME, classify_report
from src.db.models import Classification, Report, engine

DELAY_SECONDS = 1.0


def main() -> None:
    with Session(engine) as session:
        reports = session.exec(select(Report)).all()
        logger.info(f"{len(reports)} rapports à classer")

        # Recupere les rapports deja classes
        existing = session.exec(select(Classification.report_id)).all()
        already_classified = set(existing)

        to_classify = [r for r in reports if r.id not in already_classified]
        logger.info(
            f"{len(to_classify)} à classer ({len(already_classified)} déjà faits)"
        )

        success = 0
        failed = 0

        for i, report in enumerate(to_classify, 1):
            logger.info(f"[{i}/{len(to_classify)}] {report.filename}")

            try:
                result = classify_report(
                    report_text=report.full_text,
                    bea_reference=report.bea_reference,
                    aircraft_registration=report.aircraft_registration,
                )

                classification = Classification(
                    report_id=report.id,
                    report_filename=report.filename,
                    domains=json.dumps([d.value for d in result.domains]),
                    criticality=result.criticality.value,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    model_name=MODEL_NAME,
                )
                session.add(classification)
                session.commit()

                logger.success(
                    f"  → {[d.value for d in result.domains]} | "
                    f"{result.criticality.value} | conf={result.confidence:.2f}"
                )
                success += 1

            except Exception as e:
                logger.error(f"  Échec : {e}")
                session.rollback()
                failed += 1

            time.sleep(DELAY_SECONDS)

        logger.success(f"Terminé : {success} classifiés, {failed} échecs")


if __name__ == "__main__":
    main()