"""BEA scraper : télécharge les PDFs via les pages de bilans annuels par catégorie."""
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

BASE_URL = "https://bea.aero"
OUTPUT_DIR = Path("data/raw")
DELAY_SECONDS = 1.0
MAX_REPORTS = 100

HEADERS = {"User-Agent": "BEA-Research-Bot/1.0"}

BILAN_URLS = [
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/avions-legers-2023/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/avions-legers-2024/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/helicopteres-2023/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/helicopteres-2024/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/helicopteres-2025/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/ulm-2023/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/ulm-2024/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/planeurs-2024/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/ballons-2023/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/ballons-2024/",
    "https://bea.aero/accidentologie/enseignements-de-securite-aviation-legere/ballons-2025/",
]


def get_page(url: str) -> BeautifulSoup | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        logger.warning(f"Échec GET {url}: {e}")
        return None


def find_detail_links(soup: BeautifulSoup) -> list[str]:
    urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/detail/" in href and "evenements-notifies" in href:
            absolute = urljoin(BASE_URL, href)
            if absolute not in urls:
                urls.append(absolute)
    return urls


def find_bea_pdf_links(soup: BeautifulSoup) -> list[str]:
    urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf") and "bea.aero" in urljoin(BASE_URL, href):
            absolute = urljoin(BASE_URL, href)
            if absolute not in urls:
                urls.append(absolute)
    return urls


def download_pdf(url: str, output_dir: Path) -> bool:
    filename = url.split("/")[-1]
    output_path = output_dir / filename

    if output_path.exists():
        return False

    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        size_kb = len(response.content) // 1024
        logger.success(f"Telecharge : {filename} ({size_kb} KB)")
        return True
    except Exception as e:
        logger.error(f"Echec {url}: {e}")
        return False


def collect_detail_urls() -> list[str]:
    all_urls = []
    for bilan_url in BILAN_URLS:
        logger.info(f"Exploration : {bilan_url.split('/')[-2]}")
        soup = get_page(bilan_url)
        time.sleep(DELAY_SECONDS)
        if not soup:
            continue
        detail_urls = find_detail_links(soup)
        new_urls = [u for u in detail_urls if u not in all_urls]
        all_urls.extend(new_urls)
        logger.info(f"  → {len(new_urls)} nouvelles fiches (total : {len(all_urls)})")
    return all_urls


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Scraping du BEA, cible : {MAX_REPORTS} rapports")

    detail_urls = collect_detail_urls()
    logger.info(f"Total : {len(detail_urls)} fiches uniques à visiter")

    new_downloads = 0
    fiches_sans_pdf = 0

    for i, detail_url in enumerate(detail_urls, 1):
        if new_downloads >= MAX_REPORTS:
            logger.info(f"Objectif atteint : {MAX_REPORTS} nouveaux telechargements")
            break

        soup = get_page(detail_url)
        time.sleep(DELAY_SECONDS)

        if not soup:
            continue

        pdf_urls = find_bea_pdf_links(soup)
        if not pdf_urls:
            fiches_sans_pdf += 1
            continue

        for pdf_url in pdf_urls:
            if new_downloads >= MAX_REPORTS:
                break
            if download_pdf(pdf_url, OUTPUT_DIR):
                new_downloads += 1

        if i % 20 == 0:
            logger.info(f"Avancement : {i}/{len(detail_urls)} fiches, "
                        f"{new_downloads} nouveaux PDFs, "
                        f"{fiches_sans_pdf} fiches sans PDF")

    total_files = len(list(OUTPUT_DIR.glob("*.pdf")))
    logger.success(
        f"Termine : {new_downloads} nouveaux telechargements, "
        f"{fiches_sans_pdf} fiches sans PDF, "
        f"{total_files} PDFs au total dans {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()