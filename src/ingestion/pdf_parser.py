"""Extraction de texte et de métadonnées depuis les PDFs du BEA."""
import re
from pathlib import Path
import fitz
from loguru import logger


BEA_REFERENCE_PATTERN = re.compile(r"BEA\s*\d{4}[-\s]?\d{4}", re.IGNORECASE)
REGISTRATION_PATTERN = re.compile(r"\b([A-Z]{1,2}-[A-Z0-9]{3,5})\b")
DATE_PATTERN = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})")


def extract_text(pdf_path: Path) -> tuple[str, int]:
    doc = fitz.open(pdf_path)
    text_parts = [page.get_text() for page in doc]
    page_count = len(doc)
    doc.close()
    return "\n".join(text_parts), page_count


def extract_bea_reference(text: str) -> str | None:
    match = BEA_REFERENCE_PATTERN.search(text)
    if match:
        return re.sub(r"\s+", "", match.group(0)).upper()
    return None


def extract_registration(text: str) -> str | None:
    header = text[:2000]
    excluded = {"BEA", "OACI", "DGAC", "EASA", "METAR", "TAF", "CAVOK"}

    for match in REGISTRATION_PATTERN.finditer(header):
        candidate = match.group(1)
        if candidate not in excluded and len(candidate) >= 5:
            return candidate
    return None


def extract_event_date(text: str) -> str | None:
    search_zone = text[:3000]
    match = DATE_PATTERN.search(search_zone)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def extract_title(text: str) -> str | None:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return " - ".join(lines[:3])[:200]
    return None


def parse_pdf(pdf_path: Path) -> dict:
    logger.info(f"Parsing : {pdf_path.name}")

    text, page_count = extract_text(pdf_path)

    return {
        "filename": pdf_path.name,
        "file_path": str(pdf_path),
        "bea_reference": extract_bea_reference(text),
        "aircraft_registration": extract_registration(text),
        "event_date": extract_event_date(text),
        "title": extract_title(text),
        "full_text": text,
        "page_count": page_count,
    }