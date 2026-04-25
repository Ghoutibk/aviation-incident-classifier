"""Application Streamlit : interface utilisateur à 5 onglets pour démo.

Architecture :
- Onglet 1 "Upload & Classification" : colle un texte, obtient classification + HFACS
- Onglet 2 "Explorateur" : table filtrable de tous les rapports en base
- Onglet 3 "Chat RAG" : pose des questions au corpus, réponse sourcée
- Onglet 4 "Signaux faibles" : visualisation UMAP des clusters
- Onglet 5 "Veille réglementaire" : feed des alertes EASA

Pourquoi Streamlit et pas React/FastAPI direct :
- Streamlit : UI en 50 lignes de Python, parfait pour démo / POC
- Pas de frontend séparé à builder
- Idéal pour montrer un projet data/ML à des recruteurs
"""
import sys
from pathlib import Path

# CRITICAL: Add project root to sys.path BEFORE any other imports
# This ensures Streamlit subprocess can find 'src' module
_SCRIPT_PATH = Path(__file__).resolve()
_APP_DIR = _SCRIPT_PATH.parent
_PROJECT_ROOT = _APP_DIR.parent
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

import json
import os
import pandas as pd
import streamlit as st
from sqlmodel import Session, select

from src.classification.classifier import classify_report
from src.db.models import Classification, RegulatoryAlert, Report, engine, init_db
from src.extraction.hfacs_extractor import extract_factors
from src.rag.chain import ask as rag_ask
from src.weak_signals.visualization import build_umap_figure


# Config globale de la page
st.set_page_config(
    page_title="Aviation Incident Classifier",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Aviation Incident Classifier")
st.caption("Classification LLM de rapports BEA + RAG + détection de signaux faibles")

# Initialise la base locale si elle n'existe pas encore.
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Onglets
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Classification",
    "📊 Explorateur",
    "💬 Chat RAG",
    "🔍 Signaux faibles",
    "📡 Veille réglementaire",
])


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 1 : Upload & Classification
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Analyse d'un rapport")
    st.write("Colle le texte d'un rapport BEA pour obtenir une classification et une extraction HFACS.")

    col1, col2 = st.columns([2, 1])
    with col1:
        bea_ref = st.text_input("Référence BEA (optionnel)", placeholder="BEA2024-0123")
    with col2:
        immat = st.text_input("Immatriculation (optionnel)", placeholder="F-XXXX")

    report_text = st.text_area("Texte du rapport", height=300, max_chars=30000)

    col_a, col_b = st.columns(2)
    with col_a:
        do_classify = st.button("🏷️ Classifier", use_container_width=True)
    with col_b:
        do_extract = st.button("🔬 Extraire HFACS", use_container_width=True)

    if do_classify and report_text:
        with st.spinner("Analyse en cours via Mistral..."):
            result = classify_report(report_text, bea_ref or None, immat or None)
        st.success("Classification terminée")

        col1, col2, col3 = st.columns(3)
        col1.metric("Criticité", result.criticality.value.upper())
        col2.metric("Confiance", f"{result.confidence:.0%}")
        col3.metric("Domaines", len(result.domains))

        st.write("**Domaines identifiés :**")
        for d in result.domains:
            st.write(f"- `{d.value}`")

        st.write("**Raisonnement :**")
        st.info(result.reasoning)

    if do_extract and report_text:
        with st.spinner("Extraction HFACS en cours..."):
            factors = extract_factors(report_text, bea_ref or None, immat or None)
        st.success("Extraction terminée")

        st.write(f"**Cause principale** : {factors.primary_cause}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**🎯 Unsafe Acts (Niveau 1)**")
            for act in factors.unsafe_acts:
                st.write(f"- [{act.category}] {act.description}")

            st.write("**⚙️ Preconditions (Niveau 2)**")
            for p in factors.preconditions:
                st.write(f"- [{p.category}] {p.description}")
        with col2:
            st.write("**👥 Unsafe Supervision (Niveau 3)**")
            for s in factors.unsafe_supervision:
                st.write(f"- {s.description}")

            st.write("**🏢 Organizational Influences (Niveau 4)**")
            for o in factors.organizational_influences:
                st.write(f"- {o.description}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**🔧 Facteurs techniques**")
            for t in factors.technical_factors:
                st.write(f"- {t}")
        with col2:
            st.write("**🌦️ Facteurs environnementaux**")
            for e in factors.environmental_factors:
                st.write(f"- {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 2 : Explorateur
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Explorateur de rapports classifiés")

    with Session(engine) as session:
        reports = session.exec(select(Report)).all()
        classifications = session.exec(select(Classification)).all()

    # Joindre rapport + classification en DataFrame pandas
    classif_by_rid = {c.report_id: c for c in classifications}
    rows = []
    for r in reports:
        c = classif_by_rid.get(r.id)
        rows.append({
            "Fichier": r.filename,
            "Réf BEA": r.bea_reference or "",
            "Immat": r.aircraft_registration or "",
            "Pages": r.page_count,
            "Domaines": ", ".join(json.loads(c.domains)) if c else "",
            "Criticité": c.criticality if c else "",
            "Confiance": f"{c.confidence:.0%}" if c else "",
        })
    df = pd.DataFrame(rows)

    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        crit_filter = st.multiselect(
            "Filtrer par criticité",
            options=["minor", "major", "catastrophic"],
            default=[],
        )
    with col2:
        search = st.text_input("Recherche (référence / fichier)")

    filtered = df.copy()
    if crit_filter:
        filtered = filtered[filtered["Criticité"].isin(crit_filter)]
    if search:
        mask = filtered["Fichier"].str.contains(search, case=False, na=False) | \
               filtered["Réf BEA"].str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    st.write(f"**{len(filtered)} rapports affichés** (sur {len(df)} au total)")
    st.dataframe(filtered, use_container_width=True, height=500)

    # Stats rapides
    if len(df) > 0:
        st.subheader("Statistiques globales")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total rapports", len(df))
        col2.metric("Catastrophic", (df["Criticité"] == "catastrophic").sum())
        col3.metric("Major", (df["Criticité"] == "major").sum())


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3 : Chat RAG
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Pose une question au corpus BEA")
    st.write("Utilise la recherche sémantique ChromaDB + Mistral pour répondre en citant les rapports sources.")

    # Historique en session state (évite de perdre au refresh)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input(
        "Ta question",
        placeholder="Ex: Quels sont les risques liés au givrage moteur en croisière ?",
    )

    col1, col2 = st.columns([4, 1])
    with col2:
        k = st.slider("Nb sources", 3, 10, 5)

    if st.button("Poser la question", use_container_width=True) and question:
        with st.spinner("Recherche sémantique + génération..."):
            response = rag_ask(question, k=k)
        st.session_state.chat_history.append({"q": question, "response": response})

    # Affichage de l'historique (plus récent en haut)
    for exchange in reversed(st.session_state.chat_history):
        with st.container(border=True):
            st.write(f"**❓ {exchange['q']}**")
            st.write(exchange["response"].answer)

            with st.expander(f"📚 Sources ({len(exchange['response'].sources)})"):
                for s in exchange["response"].sources:
                    st.write(
                        f"**{s.bea_reference}** "
                        f"(similarité {s.similarity:.2f}) — {s.report_filename}"
                    )
                    st.caption(s.text[:300] + "...")


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 4 : Signaux faibles
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.header("Détection de signaux faibles")
    st.write("Clustering HDBSCAN sur embeddings, projection UMAP 2D. Les clusters avec beaucoup de rapports récents (2024+) sont des signaux faibles potentiels.")

    with st.spinner("Calcul des clusters et projection UMAP..."):
        fig = build_umap_figure()
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Chaque point est un rapport. Les couleurs correspondent aux clusters identifiés par HDBSCAN. Les rapports proches visuellement sont sémantiquement similaires.")


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 5 : Veille réglementaire
# ─────────────────────────────────────────────────────────────────────────────

with tab5:
    st.header("Alertes réglementaires EASA")

    with Session(engine) as session:
        alerts = session.exec(select(RegulatoryAlert)).all()

    if not alerts:
        st.warning("Aucune alerte en base. Lance `uv run python -m src.regulatory.run_regulatory` pour charger les alertes.")
    else:
        st.write(f"**{len(alerts)} alertes** enregistrées")
        for a in alerts:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{a.title}**")
                    st.caption(f"Source : {a.source}")
                with col2:
                    st.link_button("🔗 Source", a.url)
                st.write(a.summary)
                themes = json.loads(a.relevance_themes)
                if themes:
                    st.write("**Thèmes :** " + " · ".join([f"`{t}`" for t in themes]))