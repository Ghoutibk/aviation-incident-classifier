"""Détection de signaux faibles par clustering HDBSCAN.

HDBSCAN plutôt que KMeans : on ne connaît pas le nombre de thèmes à l'avance,
et les rapports hors-sujet sont mieux gérés comme du bruit (label -1) que
forcés dans un cluster.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import hdbscan
import numpy as np
from loguru import logger
from sqlmodel import Session, select

from src.db.models import Report, engine
from src.vector_store.chroma_client import get_chroma_client, get_or_create_collection


@dataclass
class ReportCluster:
    """Un cluster de rapports identifié par HDBSCAN."""
    cluster_id: int                   
    report_filenames: list[str] = field(default_factory=list)
    report_refs: list[str] = field(default_factory=list)  
    years: list[int] = field(default_factory=list)       
    size: int = 0
    recent_ratio: float = 0.0  # proportion de rapports 2024+
    keywords: list[str] = field(default_factory=list)      


def aggregate_report_embeddings() -> dict[int, np.ndarray]:
    """Un rapport = plusieurs chunks, donc on moyenne pour avoir un seul vecteur par rapport."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    data = collection.get(include=["embeddings", "metadatas"])

    embeddings = np.array(data["embeddings"])
    metadatas = data["metadatas"]

    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for emb, meta in zip(embeddings, metadatas):
        grouped[meta["report_id"]].append(emb)

    report_embeddings = {
        rid: np.mean(vecs, axis=0) for rid, vecs in grouped.items()
    }
    logger.info(f"{len(report_embeddings)} rapports agrégés (moyenne de chunks)")
    return report_embeddings


def cluster_reports(min_cluster_size: int = 3) -> list[ReportCluster]:
    """min_cluster_size=3 est calibré pour un corpus ~100 rapports — ajuster si le volume change."""
    report_embeddings = aggregate_report_embeddings()
    if not report_embeddings:
        logger.warning("Aucun embedding disponible : clustering ignoré")
        return []

    if len(report_embeddings) < min_cluster_size:
        logger.warning(
            f"Pas assez de rapports vectorisés pour HDBSCAN : {len(report_embeddings)} < {min_cluster_size}"
        )
        return []

    report_ids = list(report_embeddings.keys())
    X = np.array([report_embeddings[rid] for rid in report_ids])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(f"HDBSCAN : {n_clusters} clusters trouvés, {n_noise} points bruit")

    clusters_dict: dict[int, ReportCluster] = defaultdict(
        lambda: ReportCluster(cluster_id=-999)
    )

    with Session(engine) as session:
        for report_id, label in zip(report_ids, labels):
            report = session.get(Report, report_id)
            if not report:
                continue

            if label not in clusters_dict:
                clusters_dict[int(label)] = ReportCluster(cluster_id=int(label))

            cluster = clusters_dict[int(label)]
            cluster.report_filenames.append(report.filename)
            cluster.report_refs.append(report.bea_reference or "unknown")

            # Extraire l'année de la référence BEA (ex: "BEA2023-0375" → 2023)
            if report.bea_reference and report.bea_reference.startswith("BEA"):
                try:
                    year = int(report.bea_reference[3:7])
                    cluster.years.append(year)
                except ValueError:
                    pass

    result = []
    for label, cluster in clusters_dict.items():
        cluster.size = len(cluster.report_filenames)
        # 2024+ comme proxy d'émergence — à ajuster selon l'horizon d'analyse
        if cluster.years:
            recent_count = sum(1 for y in cluster.years if y >= 2024)
            cluster.recent_ratio = recent_count / len(cluster.years)
        result.append(cluster)

    result.sort(key=lambda c: (-c.recent_ratio, -c.size))
    return result


def detect_weak_signals(
    min_cluster_size: int = 3,
    min_recent_ratio: float = 0.5,
) -> list[ReportCluster]:
    
    all_clusters = cluster_reports(min_cluster_size=min_cluster_size)
    signals = [
        c for c in all_clusters
        if c.cluster_id != -1
        and c.recent_ratio >= min_recent_ratio
        and c.size >= min_cluster_size
    ]
    logger.success(
        f"{len(signals)} signaux faibles détectés "
        f"(clusters récents à ratio ≥ {min_recent_ratio})"
    )
    return signals


if __name__ == "__main__":
    signals = detect_weak_signals()
    for s in signals:
        print(f"\n🚨 Cluster {s.cluster_id} : {s.size} rapports, "
              f"{s.recent_ratio:.0%} récents")
        print(f"   Rapports : {s.report_refs[:5]}...")