"""BEA report scraper : télécharge les PDFs de rapports depuis bea.aero."""
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

BASE_URL = "https://bea.aero"

LIST_URLS = [
    "https://bea.aero/les-enquetes/derniers-rapports-aviation-generale/",
    "https://bea.aero/enquetes-de-securite/rapports-bea-transport-commercial/",
]

OUTPUT_DIR = Path("data/raw")
DELAY_SECONDS = 1.5
MAX_REPORTS = 50

HEADERS = {
    "User-Agent": "BEA-Research-Bot/1.0 (CV project)"
}


def get_page(url: str) -> BeautifulSoup | None:
    logger.info(f"GET {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.HTTPError as e:
        logger.warning(f"HTTP {e.response.status_code} pour {url}")
        return None
    except Exception as e:
        logger.error(f"Erreur pour {url}: {e}")
        return None


def find_detail_links(soup: BeautifulSoup) -> list[str]:
    detail_urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/detail/" in href and "evenements-notifies" in href:
            absolute_url = urljoin(BASE_URL, href)
            if absolute_url not in detail_urls:
                detail_urls.append(absolute_url)
    return detail_urls


def find_pdf_links(soup: BeautifulSoup) -> list[str]:
    pdf_urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf"):
            absolute_url = urljoin(BASE_URL, href)
            pdf_urls.append(absolute_url)
    return pdf_urls


def download_pdf(url: str, output_dir: Path) -> bool:
    filename = url.split("/")[-1]
    output_path = output_dir / filename
    if output_path.exists():
        logger.debug(f"Déjà présent : {filename}")
        return True
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        size_kb = len(response.content) // 1024
        logger.success(f"Téléchargé : {filename} ({size_kb} KB)")
        return True
    except Exception as e:
        logger.error(f"Échec {url} : {e}")
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Scraping du BEA, max {MAX_REPORTS} rapports")

    all_detail_urls = []
    for list_url in LIST_URLS:
        soup = get_page(list_url)
        if soup:
            detail_urls = find_detail_links(soup)
            logger.info(f"{len(detail_urls)} pages de détail trouvées sur {list_url}")
            all_detail_urls.extend(detail_urls)
        time.sleep(DELAY_SECONDS)

    all_detail_urls = list(set(all_detail_urls))
    logger.info(f"Total : {len(all_detail_urls)} pages de détail uniques")

    downloaded = 0
    for detail_url in all_detail_urls:
        if downloaded >= MAX_REPORTS:
            break

        soup = get_page(detail_url)
        time.sleep(DELAY_SECONDS)

        if not soup:
            continue

        pdf_urls = find_pdf_links(soup)
        if not pdf_urls:
            logger.debug(f"Aucun PDF sur {detail_url}")
            continue

        for pdf_url in pdf_urls[:1]:
            if download_pdf(pdf_url, OUTPUT_DIR):
                downloaded += 1
            time.sleep(DELAY_SECONDS)

    logger.success(f"Terminé : {downloaded} PDFs dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()