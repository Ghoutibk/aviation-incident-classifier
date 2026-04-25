"""Script d'indexation : lit les rapports SQLite, les chunke, embed, stocke dans Chroma.

Workflow :
1. Charger les rapports depuis la table `report` (SQLite)
2. Pour chaque rapport, le chunker en morceaux de ~500 tokens
3. Calculer l'embedding de chaque chunk (via sentence-transformers)
4. Stocker dans Chroma avec les métadonnées (pour citer les sources au RAG)

Idempotence : on vérifie via les IDs que les chunks ne sont pas déjà indexés.
"""
from loguru import logger
from sqlmodel import Session, select

from src.db.models import Report, engine
from src.vector_store.chroma_client import (
    get_chroma_client,
    get_embeddings,
    get_or_create_collection,
)
from src.vector_store.chunker import chunk_report


def build_chunk_id(report_id: int, chunk_index: int) -> str:
    """Génère un ID unique pour chaque chunk : 'report_42_chunk_7'.

    Chroma exige des IDs uniques par chunk dans la collection.
    """
    return f"report_{report_id}_chunk_{chunk_index}"


def main() -> None:
    # Charger les dépendances : embeddings (lent au 1er appel) + Chroma
    logger.info("Chargement du modèle d'embeddings (peut prendre 30s au 1er lancement)...")
    embedder = get_embeddings()

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Récupérer les IDs déjà indexés pour l'idempotence
    existing_ids = set(collection.get()["ids"])
    logger.info(f"{len(existing_ids)} chunks déjà indexés dans Chroma")

    with Session(engine) as session:
        reports = session.exec(select(Report)).all()
        logger.info(f"{len(reports)} rapports à traiter")

        total_chunks_new = 0

        for i, report in enumerate(reports, 1):
            # Étape 1 : chunker le rapport
            chunks = chunk_report(
                report_id=report.id,
                report_filename=report.filename,
                bea_reference=report.bea_reference,
                full_text=report.full_text,
            )

            # Étape 2 : filtrer les chunks déjà indexés
            new_chunks = [
                c for c in chunks
                if build_chunk_id(c.report_id, c.chunk_index) not in existing_ids
            ]
            if not new_chunks:
                logger.debug(f"[{i}/{len(reports)}] {report.filename} : déjà indexé")
                continue

            # Étape 3 : calculer les embeddings en batch (bien plus rapide)
            texts = [c.text for c in new_chunks]
            embeddings = embedder.embed_documents(texts)

            # Étape 4 : insérer dans Chroma
            # Chroma exige 4 listes parallèles : ids, embeddings, documents, metadatas
            ids = [build_chunk_id(c.report_id, c.chunk_index) for c in new_chunks]
            metadatas = [
                {
                    "report_id": c.report_id,
                    "report_filename": c.report_filename,
                    "bea_reference": c.bea_reference,
                    "chunk_index": c.chunk_index,
                }
                for c in new_chunks
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total_chunks_new += len(new_chunks)
            logger.info(
                f"[{i}/{len(reports)}] {report.filename} : "
                f"{len(new_chunks)} nouveaux chunks indexés"
            )

        logger.success(
            f"Terminé : {total_chunks_new} nouveaux chunks, "
            f"total collection : {collection.count()}"
        )


if __name__ == "__main__":
    main()