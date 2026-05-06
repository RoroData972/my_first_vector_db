import streamlit as st
from rag import answer_question

st.set_page_config(page_title="RAG Films V2", page_icon="🎬")

st.title("🎬 RAG Films V2")

st.markdown("Recommandation de films via RAG + Groq + embeddings.")

question = st.text_input(
    "Pose une question sur les films"
)

if question:
    with st.spinner("Recherche des meilleurs films..."):
        response = answer_question(question)

    st.success("Résultats trouvés ✅")

    st.write(response)