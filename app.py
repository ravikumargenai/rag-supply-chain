import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# ==================== CONFIG ====================
LLM_MODEL = "llama3.2"  # or "llama3.1:8b", "mistral", "phi3:medium" if you pulled them

# ==================== RAG SETUP ====================
@st.cache_resource
def load_rag_chain():
    loader = TextLoader("supply_chain_data.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = Ollama(model=LLM_MODEL, temperature=0.35, num_ctx=8192)  # higher context, lower temp for accuracy

    prompt = PromptTemplate.from_template(
        """You are an expert supply chain & logistics consultant with deep knowledge of AI applications.
        Answer the question based ONLY on the provided context. Be concise, professional, and accurate.
        If the context lacks information, say: "I don't have sufficient information from the knowledge base to answer this."

        Context:
        {context}

        Question: {question}

        Answer in clear, structured format:"""
    )

    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Load chain once
rag_chain = load_rag_chain()

# ==================== STREAMLIT UI – Professional Look ====================
st.set_page_config(page_title="Supply Chain AI Assistant", page_icon="🚚", layout="wide")

st.title("Supply Chain AI Assistant – Local RAG Agent")
st.markdown("""
Ask anything about supply chain, logistics, inventory, forecasting, risk management...  
**Fully local** (Ollama + LangChain + FAISS) — no cloud, no API keys, private & fast.
""")

# Chat-like interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your supply chain question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# import os
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st

# # ==================== CONFIG ====================
# LLM_MODEL = "llama3.2"  # or "phi3:mini", "mistral" – whichever you pulled in Ollama

# # ==================== RAG SETUP (Modern LCEL style – no deprecated RetrievalQA) ====================

# # 1. Load knowledge base
# loader = TextLoader("supply_chain_data.txt")
# documents = loader.load()

# # 2. Split into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
# texts = text_splitter.split_documents(documents)

# # 3. Embeddings (local, fast)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # 4. Vector store
# vector_store = FAISS.from_documents(texts, embeddings)
# retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # top 4 chunks

# # 5. Local LLM via Ollama (no internet, no token needed)
# llm = Ollama(model=LLM_MODEL, temperature=0.4)

# # 6. Prompt template
# prompt = PromptTemplate.from_template(
#     """You are a helpful supply chain expert. Use the following context to answer the question accurately and concisely. 
#     If the context doesn't contain the answer, say "I don't have enough information from the knowledge base."

#     Context: {context}

#     Question: {question}

#     Answer:"""
# )

# # 7. Modern LCEL RAG chain (recommended in 2026)
# rag_chain = (
#     {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
#      "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # ==================== STREAMLIT UI ====================

# st.set_page_config(page_title="Supply Chain RAG Demo", layout="wide")

# st.title("Supply Chain RAG Q&A – Fully Local & Private")
# st.markdown("""
# Ask questions about supply chain, logistics, inventory, risks...  
# Answers are **grounded** in your knowledge base → minimal hallucinations.
# """)

# question = st.text_input("Your Question:", placeholder="e.g., How to reduce supply chain risks? What is bullwhip effect?")

# if question:
#     with st.spinner("Thinking... (Ollama running locally)"):
#         answer = rag_chain.invoke(question)
    
#     st.success("Answer:")
#     st.markdown(answer)

#     st.info("This demo runs **100% locally** using Ollama LLM + FAISS vector store. No cloud/API calls after model download.")