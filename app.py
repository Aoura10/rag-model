from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load the document
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages.")

# 2. Split the text into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
chunks = text_splitter.split_documents(documents)

print(f"Split document into {len(chunks)} chunks.")
print(f"Preview of first chunk:\n{chunks[0].page_content[:200]}...")
