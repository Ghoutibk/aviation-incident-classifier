"""Schéma de la base SQLite."""
from datetime import datetime
from pathlib import Path
from sqlmodel import Field, SQLModel, create_engine


class Report(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    filename: str = Field(unique=True, index=True)
    file_path: str

    bea_reference: str | None = Field(default=None, index=True)
    aircraft_registration: str | None = Field(default=None, index=True)
    event_date: str | None = None  #format: YYYY-MM-DD
    event_location: str | None = None

    title: str | None = None
    full_text: str
    page_count: int

    parsed_at: datetime = Field(default_factory=datetime.now)


# Config de la base
DB_PATH = Path("data/bea.db")
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, echo=False)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)


class Classification(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)

    # Lien vers le rapport
    report_id: int = Field(foreign_key="report.id", index=True)
    report_filename: str = Field(index=True)

    # Result. de classification
    domains: str  # ex: '["human_factor", "operations"]'
    criticality: str  # "minor" or "major" or "catastrophic"
    confidence: float
    reasoning: str

    # Metadonnees de tracabilite
    model_name: str
    classified_at: datetime = Field(default_factory=datetime.now)


class FactorsExtraction(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)

    report_id: int = Field(foreign_key="report.id", index=True)
    report_filename: str = Field(index=True)

    unsafe_acts: str                       
    preconditions: str                    
    unsafe_supervision: str                
    organizational_influences: str          
    technical_factors: str                 
    environmental_factors: str             

    primary_cause: str
    confidence: float

    model_name: str
    extracted_at: datetime = Field(default_factory=datetime.now)