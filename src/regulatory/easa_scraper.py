"""Scraping des Safety Information Bulletins (SIB) de l'EASA.

Stratégie : la page sib-docs/page-1 expose une table HTML propre listant
toutes les publications de sécurité (938+ documents), avec colonnes :
Number, Issued by, Issue date, Subject, Approval Holder, Attachment.

C'est beaucoup plus stable que les pages décoratives type 'newsroom' :
- Format tabulaire prévisible
- Pagination via /page-1, /page-2, etc.
- Liens directs vers les PDFs des bulletins
"""
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

BASE = "https://ad.easa.europa.eu"
# La page 1 contient les 20 plus récents bulletins
SIB_URL = f"{BASE}/sib-docs/page-1"

HEADERS = {"User-Agent": "BEA-Research-Bot/1.0 (CV project)"}
DELAY = 1.5


@dataclass
class RawAlert:
    """Une alerte brute avant analyse LLM."""
    source: str          # "EASA" (issuer = EU) ou "EASA-FSAI" (issuer = US/CA/BR)
    url: str             # URL de la fiche détail sur ad.easa.europa.eu
    title: str           # le sujet du SIB
    snippet: str         # numéro + date + sujet + Approval Holder concaténés


def fetch_easa_alerts(max_alerts: int = 15) -> list[RawAlert]:
    """Récupère les SIB EASA depuis la table de publications.

    Parse la table HTML de la page-1 (20 lignes) et extrait pour chaque ligne :
    - Number (ex: 2025-09)
    - Issuer (EU, US, CA, BR — on étiquette EASA si EU, sinon EASA-FSAI)
    - Issue date
    - Subject
    - Approval Holder
    - URL de la fiche détail

    Renvoie une liste vide si erreur (le pipeline continue malgré tout).
    """
    alerts: list[RawAlert] = []
    try:
        logger.info(f"GET {SIB_URL}")
        response = requests.get(SIB_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Trouver le tableau principal (le seul qui contient des liens /ad/)
        tables = soup.find_all("table")
        target_table = None
        for t in tables:
            if t.find("a", href=lambda h: h and "/ad/" in h):
                target_table = t
                break

        if not target_table:
            logger.warning("Aucune table de publications trouvée — structure modifiée ?")
            return []

        rows = target_table.find_all("tr")[1:]  # skip header
        logger.info(f"{len(rows)} lignes trouvées dans la table")

        for row in rows[:max_alerts]:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            # Extraction des cellules par position
            number_cell = cells[0]
            issuer = cells[1].get_text(strip=True)
            issue_date = cells[2].get_text(strip=True)
            subject = cells[3].get_text(" ", strip=True)
            approval_holder = cells[4].get_text(" ", strip=True) if len(cells) > 4 else ""

            link = number_cell.find("a", href=True)
            if not link:
                continue
            number = link.get_text(strip=True)
            detail_url = urljoin(BASE, link["href"])

            # Source : EU = vraiment EASA, sinon c'est un FSAI relayé par EASA
            source = "EASA" if issuer == "EU" else f"EASA-FSAI ({issuer})"

            snippet = (
                f"SIB {number} | publié le {issue_date} | "
                f"Sujet : {subject} | Concerne : {approval_holder}"
            )

            alerts.append(RawAlert(
                source=source,
                url=detail_url,
                title=subject[:200],
                snippet=snippet[:1000],
            ))

        logger.info(f"{len(alerts)} alertes EASA extraites")
        time.sleep(DELAY)

    except Exception as e:
        logger.warning(f"Échec scraping EASA : {e} (on continue avec une liste vide)")

    return alerts