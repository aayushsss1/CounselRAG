import os
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from get_context_kg import get_full_context
import requests
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

PROMPT_TEMPLATE = '''
You are a legal domain expert assistant. Given the following excerpts from legal documents and case texts, answer the question using only the provided context.

•⁠  If the question is simple or asks for a fact or definition, give a concise, direct answer (one or two sentences).
•⁠  If the question is broad, complex, or asks for reasoning, provide a detailed answer (about 5-6 sentences) including relevant legal reasoning and case details.
•⁠  Avoid making up any information not present in the context.
•⁠  For complex questions, first identify which retrieved excerpts are most relevant, then synthesize them into a coherent legal explanation.

---
PDF Retrieved Context:
{pdf_context}

---
Knowledge Graph Context:
{kg_context}

---
Question:
{question}

---
Answer:
'''

def build_prompt(pdf_context, kg_context, question):
    return PROMPT_TEMPLATE.format(
        pdf_context=pdf_context,
        kg_context=kg_context,
        question=question
    )

def call_groq_llm(system_prompt: str, user_prompt: str):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def get_pdf_context(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 4}, search_type="mmr")
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found"
    return "\n\n".join(doc.page_content for doc in docs)

def prompt(question, model):
    pdf_context = get_pdf_context(question)
    kg_context = get_full_context(pdf_context)

    if model == "llama3-8b-8192":
        user_prompt = build_prompt(pdf_context, kg_context, question)
        answer = call_groq_llm(
            system_prompt="You are a expert legal QA assistant",
            user_prompt=user_prompt
        )
        return answer

    elif model in ["gemma:2b", "granite3.3:2b", "gemma3:4b", "legal-qa-gemma"]:
        llm = ChatOllama(
            model=model,
            temperature=0.7,
        )
        
        qaprompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        kg_context_runnable = RunnableLambda(lambda input: get_full_context(input["pdf_context"]))

        rag_chain = (
            {
                "pdf_context": RunnablePassthrough().bind(value=pdf_context),
                "kg_context": kg_context_runnable,
                "question": RunnablePassthrough(),
            }
            | qaprompt
            | llm
            | StrOutputParser()
        )

        return rag_chain.invoke({"question": question, "pdf_context": pdf_context})
    else:
        raise ValueError(f"Unsupported model: {model}")