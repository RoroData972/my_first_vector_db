import streamlit as st
from rag import answer_question

st.set_page_config(page_title="RAG Films V2")

st.title("🎬 RAG Films V2")

question = st.text_input("Pose une question sur les films")

if question:
    with st.spinner("Recherche en cours..."):
        response = answer_question(question)

    st.write(response)