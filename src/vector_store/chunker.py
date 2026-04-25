from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str              
    report_id: int         
    report_filename: str   
    bea_reference: str    
    chunk_index: int      

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def chunk_report(
    report_id: int,
    report_filename: str,
    bea_reference: str | None,
    full_text: str,
) -> list[Chunk]:

    pieces = SPLITTER.split_text(full_text)

    chunks = []
    for i, text in enumerate(pieces):
        chunks.append(Chunk(
            text=text,
            report_id=report_id,
            report_filename=report_filename,
            bea_reference=bea_reference or "unknown",
            chunk_index=i,
        ))
    return chunks