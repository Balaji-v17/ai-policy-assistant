import faiss
import json
import os
from sentence_transformers import SentenceTransformer
import logging

class RAGSystem:
    def __init__(self):
        # Ultra-lightweight model for Render stability
        self.model_name = 'paraphrase-albert-small-v2' 
        self.index_path = 'faiss_index.bin'
        self.data_path = 'policy_data/'
        
        try:
            logging.info("Loading AI Librarian...")
            self.model = SentenceTransformer(self.model_name)
            self.index = faiss.read_index(self.index_path)
            self.documents = self._load_documents()
        except Exception as e:
            logging.error(f"Init Error: {e}")

    def _load_documents(self):
        docs = []
        if os.path.exists(self.data_path):
            for filename in sorted(os.listdir(self.data_path)):
                if filename.endswith('.json'):
                    with open(os.path.join(self.data_path, filename), 'r') as f:
                        docs.append(json.load(f).get('content', ''))
        return docs

    async def query(self, user_query: str):
        question_embedding = self.model.encode([user_query])
        distances, indices = self.index.search(question_embedding, k=3)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]
