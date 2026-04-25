"""Visualisation 2D des clusters : UMAP pour la projection, Plotly pour l'interactivité."""
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
    if not report_embeddings:
        fig = px.scatter(title="Signaux faibles")
        fig.add_annotation(
            text="Aucun embedding disponible pour le moment.<br>Lance l’indexation vectorielle pour afficher cette vue.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(template="plotly_white")
        return fig

    if len(report_embeddings) < 2:
        fig = px.scatter(title="Signaux faibles")
        fig.add_annotation(
            text="Il faut au moins 2 rapports vectorisés pour calculer une projection UMAP.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(template="plotly_white")
        return fig

    report_ids = list(report_embeddings.keys())
    X = np.array([report_embeddings[rid] for rid in report_ids])

    # cosine plutôt qu'euclidean : les embeddings de phrase sont normalisés, la direction compte plus que la norme
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer.fit_transform(X)
    logger.info("Projection UMAP calculée")

    clusters = cluster_reports()
    label_by_report = {}
    for c in clusters:
        for fname in c.report_filenames:
            label_by_report[fname] = c.cluster_id

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