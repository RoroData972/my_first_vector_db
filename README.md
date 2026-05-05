# my_first_vector_db

## Installation

Pour ceux qui n'ont pas de GPU Nvidia :

pip install -r requirements.txt --break-system-packages

## Rodrigue

## Description du projet 

Ce projet implémente un système de recommandation de films basé sur une approche **RAG (Retrieval-Augmented Generation)**.

Le système :
- transforme les films en embeddings avec SentenceTransformer
- effectue une recherche vectorielle (FAISS)
- génère une réponse à l’aide d’un LLM (Groq)

---

## Difficultés rencontrées

### Problème avec FAISS (segmentation fault)

Lors de l'utilisation de FAISS pour la recherche vectorielle (`index.search`), une erreur de type **segmentation fault** s'est produite sur macOS (Apple Silicon).

Causes :
- incompatibilité entre FAISS et certaines architectures ARM
- incompatibilité avec NumPy 2.x

---

### Problème de compatibilité NumPy

Une erreur indiquait que FAISS n’était pas compatible avec NumPy 2.x.

---

### Contournement du bug FAISS

FAISS a été conservé pour la phase d’indexation (création de la base vectorielle).

Cependant, la fonction `index.search` provoquant des crashs, une solution alternative a été mise en place :

- récupération des embeddings
- calcul manuel de similarité (produit scalaire avec numpy)
- tri des résultats pour obtenir les top-k

Avantages :
- respect du TP (utilisation de FAISS)
- stabilité sur macOS

---
### Problème de taille du dataset (limitation à 500 films)

Au début du projet, le dataset était limité à 500 films afin d'accélérer les tests :

```python

documents = documents[:500]
Cependant, cela a entraîné plusieurs problèmes :

* faible diversité des résultats
* très peu de films français disponibles
* recommandations peu pertinentes
* impression que le filtre de langue ne fonctionnait pas

Après analyse, il s’est avéré que le problème ne venait pas du code mais du manque de données.

Solution :

* suppression de la limitation
* utilisation de l’ensemble du dataset (~4800 films)

Cela a permis :

* d’augmenter la diversité des résultats
* d’améliorer la pertinence des recommandations
* de valider correctement le filtre par langue


### Problème de modèle Groq

Le modèle initial (`llama3-8b-8192`) était déprécié.

Solution :
- utilisation du modèle `llama-3.3-70b-versatile`

---

### Évolution du prompt (suppression de context.txt)

Au début du projet, un fichier `context.txt` était utilisé pour construire le prompt envoyé au modèle.

Cependant, cette approche a été abandonnée au profit d’un prompt directement intégré dans le code (`system_prompt`), car :

- elle offre un meilleur contrôle sur le comportement du modèle
- elle est plus adaptée au cas d’usage (recommandation de films)
- elle simplifie l’architecture du projet

Le fichier `context.txt` a donc été supprimé dans la version finale.

---

## Solutions apportées

### Correction NumPy

pip install "numpy<2"

---

## Résultat

* Q1 → “j’ai structuré les données en texte avec titre, genres, résumé”
* Q2 → “json.loads pour parser les genres”
* Q3 → “sauvegarde FAISS pour éviter recalcul”
* Q4 → “prompt contrôlé pour éviter hallucination”
* Q5 → “réponse ‘Je ne sais pas’”

Le système permet :

- de recommander des films à partir d’une requête utilisateur
- de générer des réponses pertinentes basées sur le contexte
- d’éviter les hallucinations grâce à la contrainte "Je ne sais pas"

---

## Exemple

Requête :

film similaire à Inception

Réponse :

- recommandations de films similaires
- description, genres et notes

## Dataset

Le dataset n’est pas inclus dans le repository.

Pour exécuter le projet, placer le fichier CSV dans le dossier `Data/`.

Exemple :

Data/movies.csv

