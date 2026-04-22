"""Test rapide du classifier sur un seul rapport de la base."""
from sqlmodel import Session, select

from src.classification.classifier import classify_report
from src.db.models import Report, engine


def main() -> None:
    with Session(engine) as session:
        report = session.exec(select(Report)).first()
        if not report:
            print("Aucun rapport en base. Lance d'abord run_parsing.")
            return

        print(f"Test sur : {report.filename} (ref {report.bea_reference})")
        print(f"Texte : {len(report.full_text)} caracteres\n")

        result = classify_report(
            report_text=report.full_text,
            bea_reference=report.bea_reference,
            aircraft_registration=report.aircraft_registration,
        )

        print("=" * 60)
        print(f"DOMAINES : {[d.value for d in result.domains]}")
        print(f"CRITICITÉ : {result.criticality.value}")
        print(f"CONFIANCE : {result.confidence}")
        print(f"RAISONNEMENT : {result.reasoning}")
        print("=" * 60)


if __name__ == "__main__":
    main()