from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample policy data (we will replace later)
documents = [
    "Electric vehicle policy provides subsidies for EV buyers.",
    "Startup policy supports entrepreneurs with funding and tax benefits.",
    "Education policy focuses on digital learning and infrastructure."
]

# Convert documents to embeddings
doc_embeddings = model.encode(documents)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))


def search(query, top_k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = [documents[i] for i in indices[0]]
    return results
