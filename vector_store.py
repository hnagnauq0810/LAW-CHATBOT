import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import config

class TextEmbedder:
    def __init__(self, model_name=config.EMBEDDING_MODEL_NAME, device=config.DEVICE):
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        # e5 models usually need "query: " or "passage: " prefix
        # We assume dataset preparation adds "passage: " or we do it here.
        # For strictly demo purposes, we encode directly.
        # Checking if model is E5 based
        if "e5" in config.EMBEDDING_MODEL_NAME:
            texts = [f"passage: {t}" for t in texts]
            
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
        
    def encode_query(self, query: str) -> np.ndarray:
        if "e5" in config.EMBEDDING_MODEL_NAME:
            query = f"query: {query}"
        return self.model.encode([query], normalize_embeddings=True)

class VectorStore:
    def __init__(self, index_path=config.FAISS_INDEX_PATH, dim=config.EMBEDDING_DIM):
        self.index_path = str(index_path)
        self.dim = dim
        self.index = None
        self.load_or_create()

    def load_or_create(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                return
            except Exception as e:
                print(f"Error loading index: {e}, creating new one.")
        
        # Create new Index with ID filtering support
        # inner product (IP) for normalized vectors = cosine similarity
        quantizer = faiss.IndexFlatIP(self.dim)
        # IDMap2 enables add_with_ids and reconstruction
        self.index = faiss.IndexIDMap2(quantizer)
        print("Created new FAISS IndexIDMap2 (FlatIP)")

    def add_vectors(self, vectors: np.ndarray, ids: list[int]):
        """
        Add vectors with specific IDs (int64).
        """
        ids_arr = np.array(ids, dtype=np.int64)
        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs must have same length")
            
        self.index.add_with_ids(vectors, ids_arr)
        

    def remove_ids(self, ids: list[int]):
        """
        Remove vectors by ID (safe compatibility mode).
        """
        if not ids:
            return
        
        ids_arr = np.array(sorted(set(ids)), dtype=np.int64)
        try:
             self.index.remove_ids(ids_arr)
        except TypeError:
             # Fallback for some FAISS python bindings needing IDSelectorBatch
             sel = faiss.IDSelectorBatch(ids_arr.size, faiss.swig_ptr(ids_arr))
             self.index.remove_ids(sel)

    def search(self, query_vector: np.ndarray, k: int = 10):
        # query_vector shape (1, dim)
        check_dims = query_vector.shape[1]
        if check_dims != self.dim:
            raise ValueError(f"Query dim {check_dims} != Index dim {self.dim}")
            
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]

    def save(self):
        if self.index:
            faiss.write_index(self.index, self.index_path)
            print(f"Saved FAISS index to {self.index_path}")
            
    def rebuild(self, all_vectors: np.ndarray, all_ids: list[int]):
        """
        Complete rebuild (e.g. for cleaning up gaps or switching index type)
        """
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap2(quantizer)
        self.add_vectors(all_vectors, all_ids)
        self.save()
