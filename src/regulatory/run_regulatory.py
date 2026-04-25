"""Orchestration de la veille : scraping + analyse LLM + stockage."""
import json
import time

from loguru import logger
from sqlmodel import Session, select

from src.db.models import RegulatoryAlert, engine, init_db
from src.regulatory.alert_analyzer import analyze_alert
from src.regulatory.easa_scraper import fetch_easa_alerts

DELAY_SECONDS = 1.0


def main() -> None:
    init_db()
    raw_alerts = fetch_easa_alerts(max_alerts=15)

    if not raw_alerts:
        logger.warning("Aucune alerte EASA récupérée — la structure du site a peut-être changé")
        return

    with Session(engine) as session:
        existing_urls = set(session.exec(select(RegulatoryAlert.url)).all())
        new_alerts = [a for a in raw_alerts if a.url not in existing_urls]
        logger.info(f"{len(new_alerts)} nouvelles alertes à analyser")

        for i, alert in enumerate(new_alerts, 1):
            logger.info(f"[{i}/{len(new_alerts)}] {alert.title[:80]}")
            try:
                analysis = analyze_alert(alert.title, alert.snippet)

                record = RegulatoryAlert(
                    source=alert.source,
                    url=alert.url,
                    title=alert.title[:500],
                    summary=analysis.summary,
                    relevance_themes=json.dumps(analysis.themes),
                )
                session.add(record)
                session.commit()
                logger.success(f"  → thèmes : {analysis.themes}, score : {analysis.relevance_score}")
            except Exception as e:
                logger.error(f"  Échec : {e}")
                session.rollback()

            time.sleep(DELAY_SECONDS)

    logger.success("Veille terminée")


if __name__ == "__main__":
    main()