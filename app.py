from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. Loads the document
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages.")

# 2. Splits the text into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"Split document into {len(chunks)} chunks.")

# 3. Embed chunks and saving to ChromaDB
print("Generating embeddings... (this will take a few minutes for 2,200+ chunks)")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print("Successfully saved vectors to local ChromaDB.")
