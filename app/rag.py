import faiss
import json
import os
from sentence_transformers import SentenceTransformer
import logging

class RAGSystem:
    def __init__(self):
        self.model_name = 'all-MiniLM-L6-v2'
        self.index_path = 'faiss_index.bin'
        self.data_path = 'policy_data/'
        
        # We load these ONLY when the class is instantiated
        logging.info("Initializing AI Librarian...")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.index = faiss.read_index(self.index_path)
            self.documents = self._load_documents()
            logging.info("AI Librarian is ready!")
        except Exception as e:
            logging.error(f"Failed to load RAG components: {e}")
            raise e

    def _load_documents(self):
        docs = []
        for filename in os.listdir(self.data_path):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_path, filename), 'r') as f:
                    data = json.load(f)
                    # Assuming each JSON has a 'content' field
                    docs.append(data.get('content', ''))
        return docs

    async def query(self, user_query: str):
        # 1. Generate embedding for the question
        question_embedding = self.model.encode([user_query])
        
        # 2. Search FAISS index (return top 3 results)
        distances, indices = self.index.search(question_embedding, k=3)
        
        # 3. Pull the actual text from our docs
        context_chunks = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        
        # We'll pass this context to the Generator (Gemini) in the next step
        return context_chunks
