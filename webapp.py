import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 Local PDF RAG Chatbot")

# --- Sidebar for Document Upload ---
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a new PDF to the database", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Chunking and embedding document... This may take a minute."):
                # Save the uploaded file temporarily to read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_filepath = temp_file.name
                
                # Load and split the text
                loader = PyPDFLoader(temp_filepath)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(docs)
                
                # Embed and save to ChromaDB
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chroma_db")
                
                st.success(f"Successfully processed {len(chunks)} chunks!")
                os.remove(temp_filepath) # Clean up temporary file

# --- Main RAG Pipeline ---
@st.cache_resource
def load_rag_pipeline():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama3.2")
    
    template = """You are a helpful assistant for question-answering tasks. 
    Use the following retrieved context to answer the user's question. 
    If you don't know the answer, just say that you don't know. 

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = load_rag_pipeline()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})