from dotenv import load_dotenv
load_dotenv()

import chromadb

chroma_client = chromadb.Client()

collection_name = "test_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Defining my own documents

documents = [
    {"id": "doc1", "text": "Hello Ram!!"},
    {"id": "doc2", "text": "I am teaching RAG now. Currently in Basic Implementation stage."},
    {"id": "doc3", "text": "We will move into hybrid next."}
]

for doc in documents:
    collection.upsert(
        documents=[doc["text"]],
        ids=[doc["id"]]
    )

# I will now need a query 

query = "Hello Ram"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("Results from ChromaDB:")
print(results)