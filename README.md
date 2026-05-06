# my_first_vector_db

## Auteur

Rodrigue

---

# Nouveautés V2


- ajout d’une interface web avec Streamlit
- ajout des acteurs et réalisateurs dans les embeddings
- amélioration de la recherche sémantique
- fusion des datasets films + crédits
- optimisation du chargement des ressources

---

# Description

Ce projet implémente un système de recommandation de films basé sur une approche RAG (Retrieval-Augmented Generation).

Le système :

- transforme les films en embeddings avec SentenceTransformer
- effectue une recherche vectorielle
- génère une réponse avec Groq

---

# Installation

## Cloner le projet

```bash
git clone https://github.com/RoroData972/my_first_vector_db.git
cd my_first_vector_db
```

---

## Créer un environnement virtuel

```bash
python -m venv venv
```

---

## Activation

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## Installer les dépendances

### Linux / macOS

```bash
pip install -r requirements.txt --break-system-packages
```

### Windows

```bash
pip install -r requirements.txt
```

---

# Configuration

Créer un fichier `.env` à la racine du projet :

```env
GROQ_API_KEY=votre_cle_api
```

---

# Dataset

Le dataset n’est pas inclus dans le repository.

Placer le fichier CSV dans le dossier :

```text
Data/tmdb_5000_movies.csv
Data/tmdb_5000_credits.csv
```

---

# Génération de la base vectorielle

Avant d’utiliser le RAG, générer les embeddings :

```bash
python vector_db.py
```

Cette étape crée :

- les embeddings
- l’index vectoriel
- les fichiers FAISS

---

# Lancement du projet

## Interface web Streamlit

```bash
python -m streamlit run app.py
```

## Version console

```bash
python rag.py
```

---

# Utilisation

Une fois le programme lancé :

```text
🎬 RAG films prêt ! Tape 'quit' pour quitter.
```

## Exemples de questions

- films similaires à Inception
- films fantastiques
### Exemples supplementaire V2
- films avec Leonardo DiCaprio
- films réalisés par Christopher Nolan

## Pour quitter

```text
quit
```

ou

```text
exit
```

---

# Dépendances principales

- sentence-transformers
- groq
- numpy
- python-dotenv
- faiss-cpu

---

# Remarques

- Le projet a été développé sur macOS Apple Silicon.
- Une alternative à `index.search()` a été utilisée à cause de problèmes de compatibilité FAISS sur Mac ARM.
- Les embeddings et ressources sont chargés une seule fois au démarrage afin d’optimiser les performances.