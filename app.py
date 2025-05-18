import streamlit as st
import PyPDF2
from prompt import prompt
from persist import persist_data
import os
import base64
import time

def stick_header():
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: #2D3748;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 2px solid white;
                }
            </style>
        """,
        unsafe_allow_html=True
    )


with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Select a Model",
        ("llama3-8b-8192", "gemma:2b", "granite3.3:2b", "gemma3:4b", "legal-qa-gemma")
    )
    st.session_state["selected_model"] = model_choice
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        st.markdown("### PDF Preview")
        documents_dir = "Documents"
        os.makedirs(documents_dir, exist_ok=True)
        save_path = os.path.join(documents_dir, uploaded_pdf.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        with open(save_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="300" height="400" type="application/pdf">
            </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)

container = st.container()

def fake_stream(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

with container:
    st.markdown("# CounselRAG")
    stick_header()

if "kgmsg" not in st.session_state:
    st.session_state.kgmsg = []

for message in st.session_state.kgmsg:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Post your query!"):
    st.chat_message("human").markdown(question)
    st.session_state.kgmsg.append({"role": "human", "content": question})
    with st.spinner("Please wait..."):
        persist_data(documents_dir)
        answer = prompt(question, st.session_state["selected_model"])
    with st.chat_message("ai"):
        response = st.write_stream(fake_stream(answer))
    st.session_state.kgmsg.append({"role": "ai", "content": response})