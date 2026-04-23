"""Lance l'extraction HFACS sur tous les rapports en base, stocke en SQLite."""
import json
import time

from loguru import logger
from sqlmodel import Session, select

from src.db.models import FactorsExtraction, Report, engine, init_db
from src.extraction.hfacs_extractor import MODEL_NAME, extract_factors

DELAY_SECONDS = 1.0


def serialize_list(items: list) -> str:
    """Convertit une liste d'objets Pydantic (ou de strings) en JSON string.

    Pydantic v2 : model_dump() convertit un BaseModel en dict sérialisable.
    Pour les strings, on passe directement à json.dumps().
    """
    if not items:
        return "[]"
    if hasattr(items[0], "model_dump"):
        return json.dumps([item.model_dump() for item in items], ensure_ascii=False)
    return json.dumps(items, ensure_ascii=False)


def main() -> None:
    # Crée la table factorsextraction si elle n'existe pas (idempotent)
    init_db()

    with Session(engine) as session:
        reports = session.exec(select(Report)).all()
        logger.info(f"{len(reports)} rapports en base")

        # Reprise : ne refait pas l'extraction pour les rapports déjà traités
        existing_ids = set(session.exec(select(FactorsExtraction.report_id)).all())
        to_process = [r for r in reports if r.id not in existing_ids]
        logger.info(f"{len(to_process)} rapports à extraire ({len(existing_ids)} déjà faits)")

        success, failed = 0, 0

        for i, report in enumerate(to_process, 1):
            logger.info(f"[{i}/{len(to_process)}] {report.filename}")
            try:
                # Appel LLM (peut prendre 2-5s selon la taille du rapport)
                factors = extract_factors(
                    report_text=report.full_text,
                    bea_reference=report.bea_reference,
                    aircraft_registration=report.aircraft_registration,
                )

                # On stocke chaque niveau en JSON sérialisé dans sa colonne
                extraction = FactorsExtraction(
                    report_id=report.id,
                    report_filename=report.filename,
                    unsafe_acts=serialize_list(factors.unsafe_acts),
                    preconditions=serialize_list(factors.preconditions),
                    unsafe_supervision=serialize_list(factors.unsafe_supervision),
                    organizational_influences=serialize_list(factors.organizational_influences),
                    technical_factors=serialize_list(factors.technical_factors),
                    environmental_factors=serialize_list(factors.environmental_factors),
                    primary_cause=factors.primary_cause,
                    confidence=factors.confidence,
                    model_name=MODEL_NAME,
                )
                session.add(extraction)
                session.commit()
                logger.success(
                    f"  → {len(factors.unsafe_acts)} actes, "
                    f"{len(factors.preconditions)} préconditions, "
                    f"cause : {factors.primary_cause[:60]}..."
                )
                success += 1

            except Exception as e:
                logger.error(f"  Échec : {e}")
                session.rollback()
                failed += 1

            time.sleep(DELAY_SECONDS)

        logger.success(f"Terminé : {success} extraits, {failed} échecs")


if __name__ == "__main__":
    main()