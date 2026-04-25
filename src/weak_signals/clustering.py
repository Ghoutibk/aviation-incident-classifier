"""Détection de signaux faibles par clustering HDBSCAN sur embeddings.

Principe :
1. On récupère tous les embeddings des rapports depuis ChromaDB
2. On agrège les chunks par rapport (moyenne des embeddings) pour avoir
   1 vecteur par rapport (plus pertinent pour clusterer des rapports entiers)
3. On clusterise avec HDBSCAN → groupes de rapports thématiquement proches
4. On identifie les clusters 'émergents' : forte proportion de rapports récents

Pourquoi HDBSCAN et pas KMeans :
- Pas besoin de fixer k à l'avance (on ne sait pas combien de thèmes)
- Gère le bruit (rapports hors cluster labellisés -1)
- Clusters de tailles variables (thèmes rares vs fréquents)
- Hiérarchique : structure plus riche que KMeans
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
    """Récupère tous les chunks de Chroma et agrège par rapport (moyenne).

    Returns : dict {report_id: embedding_moyen_768dim}.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    data = collection.get(include=["embeddings", "metadatas"])

    embeddings = np.array(data["embeddings"])
    metadatas = data["metadatas"]

    # Regrouper les embeddings par report_id
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for emb, meta in zip(embeddings, metadatas):
        grouped[meta["report_id"]].append(emb)

    # Moyenne par rapport (centroide du rapport dans l'espace sémantique)
    report_embeddings = {
        rid: np.mean(vecs, axis=0) for rid, vecs in grouped.items()
    }
    logger.info(f"{len(report_embeddings)} rapports agrégés (moyenne de chunks)")
    return report_embeddings


def cluster_reports(min_cluster_size: int = 3) -> list[ReportCluster]:
    """Applique HDBSCAN sur les embeddings de rapports.

    min_cluster_size=3 : un 'cluster' doit contenir au moins 3 rapports,
    sinon c'est considéré comme du bruit. Valeur adaptée à notre corpus
    de ~100 rapports : trop haut = peu de clusters, trop bas = trop de bruit.
    """
    report_embeddings = aggregate_report_embeddings()
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

    # Enrichir chaque cluster avec les métadonnées des rapports
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

    # Calcul de métriques par cluster
    result = []
    for label, cluster in clusters_dict.items():
        cluster.size = len(cluster.report_filenames)
        # Ratio de rapports 'récents' (2024+) = signal d'émergence
        if cluster.years:
            recent_count = sum(1 for y in cluster.years if y >= 2024)
            cluster.recent_ratio = recent_count / len(cluster.years)
        result.append(cluster)

    # Trier: clusters récents puis grands clusters
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