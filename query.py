from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Connect to the existing local database
print("Connecting to ChromaDB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Setup the generative AI model
llm = OllamaLLM(model="llama3.2")

# 3. Define the instructions for the AI
template = """You are a helpful assistant for question-answering tasks. 
Use the following retrieved context to answer the user's question. 
If you don't know the answer, just say that you don't know. 

Context: {context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Function to format retrieved chunks into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Create the final RAG pipeline using modern LCEL architecture
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Ask a question!
query = "Who is the author of this document?"
print(f"\nQuestion: {query}")
print("Thinking...")

response = rag_chain.invoke(query)
print(f"\nAnswer: {response}")