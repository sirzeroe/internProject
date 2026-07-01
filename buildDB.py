import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load and prep the data
print("Loading CSV...")
df = pd.read_csv('data/ticketData.csv')
# Keep only the essential columns and drop missing data
df = df[['body', 'answer']].dropna()

# 2. Convert rows into LangChain Document objects
print("Creating document chunks...")
docs = []
for _, row in df.iterrows():
    # Combining the problem and the solution so the AI has the full context
    content = f"Issue: {row['body']}\nResolution: {row['answer']}"
    docs.append(Document(page_content=content))

# 3. Initialize the local embedding model
print("Initializing embedding model...")
# This downloads the ~90MB all-MiniLM-L6-v2 model automatically on the first run
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Build and persist the local Chroma database
print("Building ChromaDB vector store...")
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print(f"Success! Vectorized {len(docs)} tickets and saved to disk.")