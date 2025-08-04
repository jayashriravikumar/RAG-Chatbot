import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

file_path = "data/paul_graham_essay.txt"
loader = TextLoader(file_path)

documents = loader.load()
print(f"Document loaded. Total pages/docs: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)
print(f"Document split into {len(chunks)} chunks.")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

print("Creating vector store... This may take a moment.")
db = FAISS.from_documents(chunks, embeddings)
print("Vector store created successfully.")

db.save_local("faiss_index")
print("FAISS index saved locally.")

query = "What did the author do during his time at Y Combinator?"
print(f"\nPerforming similarity search for query: '{query}'")

docs = db.similarity_search(query, k=3)

print("\n--- Top 3 Relevant Chunks ---")
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print("---------------------\n")
