"""Parse tous les PDFs de data/raw/ et les stocke en SQLite."""
from pathlib import Path

from loguru import logger
from sqlmodel import Session, select

from src.db.models import Report, engine, init_db
from src.ingestion.pdf_parser import parse_pdf

RAW_DIR = Path("data/raw")


def main() -> None:
    init_db()
    logger.info("Base de donnees initialisee")

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    logger.info(f"{len(pdf_files)} PDFs à parser")

    inserted = 0
    skipped = 0
    failed = 0

    with Session(engine) as session:
        for pdf_path in pdf_files:
            stmt = select(Report).where(Report.filename == pdf_path.name)
            if session.exec(stmt).first():
                logger.debug(f"Deja en base : {pdf_path.name}")
                skipped += 1
                continue

            try:
                data = parse_pdf(pdf_path)
                report = Report(**data)
                session.add(report)
                session.commit()
                inserted += 1
                logger.success(
                    f"✓ {pdf_path.name} | "
                    f"ref={data['bea_reference']} | "
                    f"immat={data['aircraft_registration']} | "
                    f"pages={data['page_count']}"
                )
            except Exception as e:
                logger.error(f"Echec sur {pdf_path.name} : {e}")
                session.rollback()
                failed += 1

    logger.success(
        f"Termine : {inserted} inseres, {skipped} deja presents, {failed} echecs"
    )


if __name__ == "__main__":
    main()