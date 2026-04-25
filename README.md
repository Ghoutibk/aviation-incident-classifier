# ✈️ Aviation Incident Classifier

Plateforme d'intelligence sécurité aérienne pour l'analyse automatisée de rapports d'incidents du BEA (Bureau d'Enquêtes et d'Analyses). Le système combine classification par LLM, extraction structurée de facteurs HFACS, retrieval sémantique via ChromaDB, détection de signaux faibles, et veille réglementaire EASA.

**Stack** : Python 3.11 • Mistral (small-latest) • LangChain • ChromaDB • HDBSCAN • UMAP • FastAPI • Streamlit • SQLModel

---

## 🎯 Objectifs

Transformer 100+ rapports PDF du BEA en une base de connaissances interrogeable, avec :

- **Classification multi-label** par domaine de risque (6 domaines × 3 criticités) via Mistral + sortie structurée Pydantic
- **Extraction des facteurs contributifs** selon le modèle HFACS (Human Factors Analysis and Classification System)
- **Chatbot RAG** pour interroger le corpus en langage naturel avec citation des sources
- **Détection de signaux faibles** par clustering HDBSCAN + visualisation UMAP
- **Veille réglementaire** automatisée sur les publications EASA

---

## 📊 Résultats

- 100+ rapports BEA ingérés et parsés (SQLite)
- 2 500+ chunks indexés dans ChromaDB avec embeddings multilingues
- Classification évaluée sur 20 rapports annotés manuellement : **F1 macro ~0.81** sur 6 domaines × 3 criticités
- Pipeline RAG : retrieval top-5 avec temps de réponse < 3s

---

## 🏗️ Architecture

### Vue d'ensemble du pipeline

Le système suit un pipeline linéaire en 10 étapes, depuis la collecte brute des PDFs jusqu'à l'exposition via deux interfaces :

```
PDFs BEA
   │
   ▼
[1] Scraping BEA ──────────────► src/ingestion/bea_scraper.py
   │                              Téléchargement des rapports publics BEA
   ▼
[2] Parsing PDF ───────────────► src/ingestion/pdf_parser.py
   │                              Extraction du texte brut + métadonnées
   ▼
[3] Stockage SQLite ───────────► src/db/ (SQLModel)
   │                              Tables : Report, Classification, HFACSFactor,
   │                              RegulatoryAlert
   ▼
[4] Classification LLM ────────► src/classification/
   │                              Mistral small-latest + sortie structurée Pydantic
   │                              6 domaines × 3 criticités → F1 macro ~0.81
   ▼
[5] Extraction HFACS ──────────► src/extraction/
   │                              Facteurs contributifs humains/organisationnels
   │                              structurés selon le modèle Wiegmann & Shappell
   ▼
[6] Indexation vectorielle ────► src/vector_store/indexer.py
   │                              Découpage en chunks → embeddings multilingues
   │                              → ChromaDB (2 500+ chunks, ~100 000 vecteurs max)
   ▼
[7] Veille EASA ───────────────► src/regulatory/
   │                              Scraping & stockage des alertes réglementaires
   ▼
[8] Détection signaux faibles ─► src/weak_signals/
   │                              HDBSCAN (clustering sans k fixé) + UMAP (2D)
   ▼
[9] RAG Chain ─────────────────► src/rag/
   │                              Retrieval top-5 dans ChromaDB via LangChain
   │                              → Génération de réponse par Mistral (< 3 s)
   ▼
[10] Interfaces utilisateur
      ├── Streamlit ────────────► app/streamlit_app.py  (démo interactive)
      └── FastAPI ──────────────► src/api/main.py       (REST API + /docs)
```

### Flux de données

```
Rapport PDF
    │
    ├─► Texte brut ──► SQLite (table Report)
    │                        │
    │        ┌───────────────┼───────────────┐
    │        ▼               ▼               ▼
    │   Classification   Extraction      Veille
    │   (Mistral LLM)    HFACS           EASA
    │        │               │
    │        └───────────────┘
    │                │
    │           SQLite (Classification, HFACSFactor)
    │
    ├─► Chunks (LangChain TextSplitter)
    │        │
    │        ▼
    │   Embeddings multilingues (sentence-transformers)
    │        │
    │        ▼
    │   ChromaDB (persistance sur disque)
    │        │
    │        ├─► Clustering HDBSCAN + projection UMAP (signaux faibles)
    │        │
    │        └─► RAG Chain (retrieval top-5 → Mistral → réponse citée)
    │
    └─► Interfaces : Streamlit UI  /  FastAPI REST
```

### Modules IA principaux

| Module | Technologie | Rôle |
|---|---|---|
| `classification/` | Mistral + Pydantic | Classification multi-label : 6 domaines (facteur humain, technique, météo, ATC, procédures, environnement) × 3 criticités (faible / modérée / élevée) |
| `extraction/` | Mistral + Pydantic | Extraction structurée des facteurs HFACS à 4 niveaux : Unsafe Acts → Preconditions → Supervision → Organizational |
| `vector_store/` | ChromaDB + LangChain | Indexation de 2 500+ chunks avec embeddings multilingues ; persistance fichier, pas de service externe |
| `rag/` | LangChain RetrievalQA | Retrieval sémantique top-5 puis génération par Mistral ; temps de réponse < 3 s |
| `weak_signals/` | HDBSCAN + UMAP | Clustering sans nombre de clusters fixé, projection 2D pour visualisation des thèmes émergents |
| `regulatory/` | Scraping + SQLModel | Veille automatisée sur les publications EASA, stockage et exposition via API |

### Base de données (SQLModel / SQLite)

```
Report ──────────────┐
  id, title, date,   │ 1─────N  Classification
  url, raw_text,     │            domain, criticality,
  parsed_at          │            confidence, reasoning,
                     │            model_name, classified_at
                     │
                     │ 1─────N  HFACSFactor
                                  level, category,
                                  description, extracted_at

RegulatoryAlert
  id, source, title, date, url, content
```

Chaque `Classification` conserve `model_name`, `classified_at`, `confidence` et `reasoning` pour assurer la traçabilité des prédictions (conformité AI Act Art. 12-13).

### Intégration Mistral / LangChain / ChromaDB

```python
# Chaîne RAG (src/rag/)
retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 5})
chain = RetrievalQA.from_chain_type(
    llm=ChatMistralAI(model="mistral-small-latest"),
    retriever=retriever,
)

# Classification structurée (src/classification/)
chain = prompt | llm.with_structured_output(ClassificationOutput)  # Pydantic
```

LangChain orchestre les appels LLM avec retry automatique sur JSON invalide ; ChromaDB est utilisé en mode **embedded** (un seul répertoire `data/chroma_db/`), sans service séparé à lancer.

### Interfaces utilisateur

| Interface | Commande de lancement | Fonctionnalités principales |
|---|---|---|
| **Streamlit** (`app/streamlit_app.py`) | `uv run streamlit run app/streamlit_app.py` | Chatbot RAG, visualisation UMAP des clusters, tableau de bord des classifications, alertes EASA |
| **FastAPI** (`src/api/main.py`) | `uv run uvicorn src.api.main:app --reload` | Endpoints REST pour rapports, classifications, facteurs HFACS, alertes ; documentation interactive sur `/docs` |

## 🚀 Quick Start

### Prérequis

- Python 3.11+
- [uv](https://astral.sh/uv) (gestionnaire de paquets)
- Une clé API Mistral (gratuite sur [console.mistral.ai](https://console.mistral.ai))

### Installation

```bash
# Cloner le repo
git clone https://github.com/<your-username>/aviation-incident-classifier.git
cd aviation-incident-classifier

# Créer l'environnement avec uv
uv sync

# Configurer la clé API
cp .env.example .env
# Puis édite .env et colle ta clé Mistral
```

### Pipeline complet

```bash
# 1. Scraper les rapports BEA (~15 min)
uv run python src/ingestion/bea_scraper.py

# 2. Parser les PDFs et stocker en SQLite (~2 min)
uv run python -m src.ingestion.run_parsing

# 3. Classifier tous les rapports via Mistral (~5 min)
uv run python -m src.classification.run_classification

# 4. Extraire les facteurs HFACS (~10 min)
uv run python -m src.extraction.run_extraction

# 5. Indexer dans ChromaDB (~5 min, 1er run plus long car téléchargement modèle)
uv run python -m src.vector_store.indexer

# 6. Récupérer les alertes EASA (~1 min)
uv run python -m src.regulatory.run_regulatory
```

### Lancer l'interface

```bash
# Streamlit (UI démo)
uv run streamlit run app/streamlit_app.py

# Ou FastAPI (API REST, docs sur /docs)
uv run uvicorn src.api.main:app --reload
```

---

## 🔧 Choix techniques

### Pourquoi Mistral plutôt qu'OpenAI ?

- **Souveraineté** : modèle européen, pertinent pour des données de sécurité aérienne
- **Coût** : Mistral Small à 0.20 €/M tokens vs GPT-4 à 30 €/M tokens (150× moins cher)
- **Qualité suffisante** : le français natif suffit pour cette tâche de classification

### Pourquoi HFACS pour l'extraction ?

Le modèle HFACS (Wiegmann & Shappell, 2003) est un **standard reconnu en safety aéronautique**. Les rapports BEA évoquent implicitement cette structure à 4 niveaux. Utiliser HFACS rend l'extraction :
- Plus structurée qu'une extraction libre
- Alignée avec les pratiques pro du domaine
- Interprétable pour un expert sécurité

### Pourquoi ChromaDB plutôt que pgvector / FAISS ?

Pour un POC à 2500 chunks, **ChromaDB en mode embedded** (juste un fichier) offre :
- Installation trivial : `pip install chromadb`, rien à configurer
- API Python native, intégration LangChain immédiate
- Persistance sur disque, pas de service à lancer
- Suffisant jusqu'à ~100k vecteurs

Pour du scaling production, **Qdrant** serait le choix (client-server, gRPC, replication). Pour >1M vecteurs avec PostgreSQL existant, **pgvector**.

### Pourquoi HDBSCAN pour le clustering ?

- **Pas besoin de fixer k** (contrairement à KMeans) : on ne sait pas a priori combien de thèmes existent
- **Gère le bruit** : les rapports atypiques sont labellisés `-1` au lieu d'être forcés dans un cluster
- **Clusters de tailles variables** : un thème rare (5 rapports) coexiste avec un thème dominant (40 rapports)
- Référence standard en text clustering depuis 2017

### Sortie structurée Pydantic

Chaque appel LLM retourne un objet Pydantic validé (pas un texte libre à re-parser). Avantages :
- **Fiabilité** : si le JSON est invalide, LangChain retry automatiquement
- **Typage fort** : utilisable directement dans le code Python
- **Documentation** : le schéma Pydantic sert à la fois de contrat LLM et de doc API

---

## 🛡️ Gouvernance & Explicabilité (AI Act)

Le projet intègre plusieurs principes du Règlement européen sur l'IA (AI Act) :

- **Traçabilité des prédictions** (Art. 12-13) : chaque classification stocke `model_name`, `classified_at`, `confidence` et `reasoning`
- **Transparence** : le champ `reasoning` de chaque classification explique les décisions du LLM
- **Qualité des données** (Art. 10) : le jeu d'évaluation est documenté, les annotations manuelles sont versionnées
- **Logging structuré** : tous les appels LLM sont loggés avec `loguru` pour audit

---

## 📈 Évaluation

Le classifier est évalué sur un jeu de **20 rapports annotés manuellement** (vérité terrain).

Métriques calculées :
- **Accuracy / F1 macro / F1 weighted** sur la criticité
- **F1 macro / F1 micro** sur les domaines (multi-label)
- **Précision / Rappel / F1 par classe** pour identifier les classes problématiques
- **Matrice de confusion**

Pour lancer l'évaluation :
```bash
uv run python -m src.evaluation.evaluate
```

Les résultats sont sauvegardés dans `data/annotated/evaluation_results.json`.

---

## 🗂️ Structure du projet