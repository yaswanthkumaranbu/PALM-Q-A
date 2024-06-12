import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("RBAC Q&A 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

user_id = 1
if question:
    chain = get_qa_chain(user_id)
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
 





