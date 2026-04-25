"""Visualisation 2D des clusters avec UMAP + Plotly.

UMAP : Uniform Manifold Approximation and Projection.
Projette des vecteurs haute dimension (768D) vers 2D en préservant
la structure locale (points proches en 768D restent proches en 2D).
Plus performant que t-SNE pour les grands datasets.
"""
import numpy as np
import plotly.express as px
import umap
from loguru import logger
from plotly.graph_objs import Figure
from sqlmodel import Session, select

from src.db.models import Report, engine
from src.weak_signals.clustering import aggregate_report_embeddings, cluster_reports


def build_umap_figure() -> Figure:
    """Génère un scatter plot Plotly interactif des rapports clusterisés."""
    report_embeddings = aggregate_report_embeddings()
    report_ids = list(report_embeddings.keys())
    X = np.array([report_embeddings[rid] for rid in report_ids])

    # UMAP projection
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer.fit_transform(X)
    logger.info("Projection UMAP calculée")

    # Récupère les labels de cluster + metadata
    clusters = cluster_reports()
    label_by_report = {}
    for c in clusters:
        for fname in c.report_filenames:
            label_by_report[fname] = c.cluster_id

    # Récupère les metadata des rapports
    filenames, labels, refs = [], [], []
    with Session(engine) as session:
        for rid in report_ids:
            r = session.get(Report, rid)
            if r:
                filenames.append(r.filename)
                labels.append(str(label_by_report.get(r.filename, -1)))
                refs.append(r.bea_reference or "unknown")

    fig = px.scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        color=labels,
        hover_data={"filename": filenames, "bea_ref": refs},
        title="Clustering des rapports BEA (UMAP 2D + HDBSCAN)",
        labels={"color": "Cluster", "x": "UMAP-1", "y": "UMAP-2"},
    )
    fig.update_traces(marker={"size": 10, "opacity": 0.7})
    return fig


if __name__ == "__main__":
    fig = build_umap_figure()
    fig.write_html("data/weak_signals_umap.html")
    logger.success("Visualisation sauvegardée : data/weak_signals_umap.html")