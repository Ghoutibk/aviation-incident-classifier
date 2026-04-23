"""Prépare un fichier CSV d'annotation manuelle pour l'évaluation."""
import csv
import random
from pathlib import Path

from loguru import logger
from sqlmodel import Session, select

from src.db.models import Report, engine

# Paramètres
NB_SAMPLES = 20
OUTPUT_PATH = Path("data/annotated/annotation_set.csv")
RANDOM_SEED = 42


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        all_reports = session.exec(select(Report)).all()

        # Filtre : exclut les documents qui ne sont pas des rapports BEA
        # (fiches pedagogiques FFA/DGAC ramassees par le scraper)
        reports = [
            r for r in all_reports
            if r.bea_reference  # doit avoir une référence BEA
            and not r.filename.startswith(("FFA_", "DGAC_", "Fiche_"))
        ]
        logger.info(f"{len(reports)} rapports BEA disponibles (sur {len(all_reports)})")

        # Echantillon stratifie basique : on prend un mix de tailles
        random.seed(RANDOM_SEED)
        short_reports = [r for r in reports if r.page_count < 15]
        long_reports = [r for r in reports if r.page_count >= 15]

        # 15 courts + 5 longs (si possible) pour varier
        sample = random.sample(short_reports, min(15, len(short_reports)))
        sample += random.sample(long_reports, min(5, len(long_reports)))

        logger.info(f"Échantillon : {len(sample)} rapports sélectionnés")

        # Génère le CSV d'annotation
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "filename",
                "bea_reference",
                "aircraft_registration",
                "page_count",
                "text_preview",  # aperçu pour te rappeler le contexte
                "true_domains",  # À REMPLIR : domaines séparés par |
                "true_criticality",  # À REMPLIR : minor | major | catastrophic
                "difficulty",  # À REMPLIR : easy | medium | hard
                "notes",  # À REMPLIR : commentaires libres
            ])
            for r in sample:
                preview = r.full_text[:300].replace("\n", " ").replace(";", ",")
                writer.writerow([
                    r.filename,
                    r.bea_reference or "",
                    r.aircraft_registration or "",
                    r.page_count,
                    preview,
                    "",  # true_domains — à remplir
                    "",  # true_criticality — à remplir
                    "",  # difficulty — à remplir
                    "",  # notes — à remplir
                ])

        logger.success(f"Fichier créé : {OUTPUT_PATH}")
        logger.info("")
        logger.info("MODE D'EMPLOI :")
        logger.info("1. Ouvre data/annotated/annotation_set.csv dans VS Code (ou Excel)")
        logger.info("2. Pour chaque rapport, lis le PDF (data/raw/<filename>)")
        logger.info("3. Remplis les colonnes :")
        logger.info("   - true_domains : liste des domaines séparés par |")
        logger.info("     ex: human_factor|operations")
        logger.info("   - true_criticality : minor, major ou catastrophic")
        logger.info("   - difficulty : easy, medium ou hard")
        logger.info("   - notes : tes remarques (optionnel)")
        logger.info("4. Sauvegarde et lance le script d'évaluation")


if __name__ == "__main__":
    main()