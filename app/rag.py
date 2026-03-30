import faiss
import json
import os
from sentence_transformers import SentenceTransformer
import logging

# Configure logging to see progress in Render's console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        # We define paths relative to the project root
        self.model_name = 'all-MiniLM-L6-v2'
        self.index_path = 'faiss_index.bin'
        self.data_path = 'policy_data/'
        
        logger.info("--- AI Librarian: Starting Initialization ---")
        
        try:
            # 1. Load the Embedding Model (Sentence Transformer)
            self.model = SentenceTransformer(self.model_name)
            
            # 2. Load the FAISS Index (The 'Brain' for searching)
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
            
            # 3. Load the Document Text
            self.documents = self._load_documents()
            
            logger.info(f"--- AI Librarian: Ready with {len(self.documents)} policies ---")
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load RAG components: {e}")
            raise e

    def _load_documents(self):
        """Loads all policy text from the JSON files in policy_data/"""
        docs = []
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} not found!")
            return docs

        for filename in sorted(os.listdir(self.data_path)):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.data_path, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # We extract the 'content' field from your policy JSONs
                        content = data.get('content', '')
                        if content:
                            docs.append(content)
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        return docs

    async def query(self, user_query: str):
        """The main RAG engine: Embed -> Search -> Retrieve"""
        try:
            # 1. Convert user question into a vector (embedding)
            question_embedding = self.model.encode([user_query])
            
            # 2. Search FAISS index for the top 3 most relevant matches
            # 'k=3' means we grab the three best evidence chunks
            distances, indices = self.index.search(question_embedding, k=3)
            
            # 3. Retrieve the actual text for those matches
            context_chunks = []
            for i in indices[0]:
                if i != -1 and i < len(self.documents):
                    context_chunks.append(self.documents[i])
            
            return context_chunks
            
        except Exception as e:
            logger.error(f"Query Processing Error: {e}")
            return ["Error retrieving policy context."]
