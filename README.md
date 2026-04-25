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

```
aviation-incident-classifier/
├── app/
│   └── streamlit_app.py        # Interface Streamlit (chatbot RAG, dashboard, UMAP)
├── src/
│   ├── ingestion/              # Scraping BEA + parsing PDF → SQLite
│   │   ├── bea_scraper.py
│   │   ├── pdf_parser.py
│   │   └── run_parsing.py
│   ├── db/
│   │   └── models.py           # Modèles SQLModel (Report, Classification, HFACSFactor, RegulatoryAlert)
│   ├── classification/         # Classification multi-label via Mistral + Pydantic
│   │   ├── classifier.py
│   │   ├── schemas.py
│   │   └── taxonomy.py         # 6 domaines × 3 criticités
│   ├── extraction/             # Extraction des facteurs HFACS
│   │   ├── hfacs_extractor.py
│   │   └── hfacs_schema.py
│   ├── vector_store/           # Indexation ChromaDB (2 500+ chunks, embeddings multilingues)
│   │   ├── chunker.py
│   │   ├── indexer.py
│   │   └── chroma_client.py
│   ├── rag/                    # RAG Chain LangChain + Mistral (retrieval top-5)
│   │   ├── chain.py
│   │   └── retriever.py
│   ├── weak_signals/           # Clustering HDBSCAN + visualisation UMAP
│   │   ├── clustering.py
│   │   └── visualization.py
│   ├── regulatory/             # Veille réglementaire EASA
│   │   ├── easa_scraper.py
│   │   └── alert_analyzer.py
│   ├── evaluation/             # Évaluation du classifier
│   │   └── evaluate.py
│   └── api/
│       └── main.py             # API REST FastAPI (endpoints + /docs)
├── data/
│   ├── reports/                # PDFs BEA téléchargés
│   ├── chroma_db/              # Base vectorielle ChromaDB (persistance disque)
│   └── annotated/              # Jeu d'évaluation annoté manuellement
└── main.py                     # Point d'entrée principal
```

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
