"""Schéma de la base SQLite."""
from datetime import datetime
from pathlib import Path
from sqlmodel import Field, SQLModel, create_engine


class Report(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # Metadonnees fichier
    filename: str = Field(unique=True, index=True)
    file_path: str

    # Metadonnees extr. du contenu
    bea_reference: str | None = Field(default=None, index=True)
    aircraft_registration: str | None = Field(default=None, index=True)
    event_date: str | None = None  # format ISO: YYYY-MM-DD
    event_location: str | None = None

    # Contenu
    title: str | None = None
    full_text: str
    page_count: int

    # Traçabilite
    parsed_at: datetime = Field(default_factory=datetime.now)


# Config de la base
DB_PATH = Path("data/bea.db")
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, echo=False)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)