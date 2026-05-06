from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import pandas as pd


df = pd.read_csv("data/tmdb_5000_movies.csv")
df_credits = pd.read_csv("Data/tmdb_5000_credits.csv")
df = df.merge(
    df_credits,
    left_on="id",
    right_on="movie_id"
)

print(df.columns)

print(df.head(2))

def get_embeddings(model, chunks):
    embeddings = model.encode(
        chunks,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    return index


def save_index(index, documents, path="faiss_index"):
    faiss.write_index(index, f"{path}.index")

    with open(f"{path}.json", "w") as f:
        json.dump(documents, f)


if __name__ == "__main__":
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    def parse_genres(genre_str):
        try:
            genres = json.loads(genre_str)
            return ", ".join([g["name"] for g in genres])
        except:
            return ""

    documents = []
def extract_actors(cast_json):
    try:
        cast = json.loads(cast_json)

        actors = [actor["name"] for actor in cast[:5]]

        return ", ".join(actors)

    except:
        return ""


def extract_director(crew_json):
    try:
        crew = json.loads(crew_json)

        for person in crew:
            if person["job"] == "Director":
                return person["name"]

        return ""

    except:
        return ""


for i, row in df.iterrows():
    genres = parse_genres(row["genres"])

    actors = extract_actors(row["cast"])

    director = extract_director(row["crew"])

    texte = f"""
    Titre: {row['title_x']}
    Genres: {genres}
    Réalisateur: {director}
    Acteurs: {actors}
    Date: {row['release_date']}
    Note: {row['vote_average']}
    Résumé: {row['overview']}
    """

    documents.append({
        "contenu": texte,
        "metadata": {
            "titre": row["title_x"],
            "note": row["vote_average"],
            "langue": row["original_language"]
        }
    })

    # Pour accélérer les tests, limiter le nombre de films (ex: 500)
		#documents = documents[:500]
    #df = df.sample(frac=1).reset_index(drop=True) melanger les film
    documents = documents[:5000]
    

chunks = [doc["contenu"] for doc in documents]

#  DEBUG
print("NB DOCUMENTS :", len(documents))
print("NB CHUNKS :", len(chunks))

embeddings = get_embeddings(model, chunks)

index = create_faiss_index(embeddings)

save_index(index, documents)

print("Index FAISS films sauvegardé")
