import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title("📄 Local PDF RAG Chatbot")

# Cache the database and model loading to optimize speed
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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box for the user
if user_input := st.chat_input("Ask a question about your PDF..."):
    # Display the user's question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display the AI's response
    with st.chat_message("assistant"):
        with st.spinner("Searching PDF and generating answer..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})