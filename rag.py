from groq import Groq
from dotenv import load_dotenv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
#import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()


## charger index FAIS
def load_index(path="faiss_index"):
    with open(f"{path}.json", "r") as f:
        documents = json.load(f)

    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    texts = [doc["contenu"] for doc in documents]
    embeddings = model.encode(texts, normalize_embeddings=True)

    return embeddings, documents

#******ne semble egalement pas marcher avec mon mac
# def load_index(path="faiss_index"):
#     index = faiss.read_index(f"{path}.index")

#     with open(f"{path}.json", "r") as f:
#         documents = json.load(f)

#     return index, documents




## puis la recherche

#*****Probleme avec mac M
#def retrieve(question, model, index, documents, k=3):
#    q_embedding = model.encode([question], normalize_embeddings=True)

#    scores, indices = index.search(q_embedding.astype("float32"), k)

#    results = []
#    for i, idx in enumerate(indices[0]):
#        results.append({
#            "contenu": documents[idx]["contenu"],
#            "metadata": documents[idx]["metadata"],
#            "score": float(scores[0][i])
#        })

##    return results


# def retrieve(question, model, index, documents, k=3):
#     q_embedding = model.encode([question], normalize_embeddings=True)[0]

#     all_embeddings = index.reconstruct_n(0, index.ntotal)

#     scores = np.dot(all_embeddings, q_embedding)

#     top_indices = np.argsort(scores)[::-1][:k]

#     results = []
#     for idx in top_indices:
#         results.append({
#             "contenu": documents[idx]["contenu"],
#             "metadata": documents[idx]["metadata"],
#             "score": float(scores[idx])
#         })

#     return results

# def retrieve(question, model, embeddings, documents, k=3):
#     q_embedding = model.encode([question], normalize_embeddings=True)[0]

#     scores = np.dot(embeddings, q_embedding)

#     top_indices = np.argsort(scores)[::-1][:k]

#     results = []
#     for idx in top_indices:
#         results.append({
#             "contenu": documents[idx]["contenu"],
#             "metadata": documents[idx]["metadata"],
#             "score": float(scores[idx])
#         })

#     return results
#Pas de langue

def retrieve(question, model, embeddings, documents, k=10):
    q_embedding = model.encode([question], normalize_embeddings=True)[0]

    scores = np.dot(embeddings, q_embedding)

    top_indices = np.argsort(scores)[::-1]

    results = []

    question_lower = question.lower()

    for idx in top_indices:
        doc = documents[idx]

        langue = doc["metadata"].get("langue", "")

        # 🔥 filtre français
        if "vf" in question_lower or "francais" in question_lower or "français" in question_lower:
            if langue != "fr":
                continue

        # 🔥 filtre international (tout sauf français)
        if "international" in question_lower:
            if langue == "fr":
                continue

        results.append({
            "contenu": doc["contenu"],
            "metadata": doc["metadata"],
            "score": float(scores[idx])
        })

        if len(results) >= k:
            break

    return results


# construire le contexte
def build_context(question, results):
    context = "Voici des informations issues de la base :\n\n"

    for i, res in enumerate(results):
        context += f"Chunk {i+1}:\n{res['contenu']}\n\n"

    return context


#  puis la réponse généré
def answer_question(question):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    model_embed = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    
    embeddings, documents = load_index()

    results = retrieve(question, model_embed, embeddings, documents, k=10)

    # index, documents = load_index()

    # results = retrieve(question, model_embed, index, documents)

    context = build_context(question, results)

    system_prompt = f"""

    Tu es un expert en recommandation de films.

    Tu dois proposer des films pertinents à partir des informations fournies.

    Pour chaque film :

    - Donne le titre

    - Donne la note

    - Donne les genres

    - Donne une courte description

    Format de réponse attendu :

    1. Titre (Note)

    Genres: ...

    Description: ...

    Règles :

    - N'utilise que les informations du contexte

    - Ne mentionne PAS les numéros de chunk

    - Si aucune information pertinente, "Je ne sais pas"


    {context}
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        #model="llama3-8b-8192" ce modèle n’existe plus sur Groq
        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content


# if __name__ == "__main__":
#     question = "films fantastiques en vf"
#     # Test : question hors base doit répondre "Je ne sais pas"
#     #question = "Comment s'appelle le champignon de mon gnome ?" 
#     response = answer_question(question)
#     print(response)

if __name__ == "__main__":
    print("🎬 RAG films prêt ! Tape 'quit' pour quitter.\n")

    while True:
        print("\n🟢 Pose ta question :")
        question = input("> ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        response = answer_question(question)

        print("\n🎯 Réponse :\n")
        print(response)
        print("\n" + "-"*50 + "\n")