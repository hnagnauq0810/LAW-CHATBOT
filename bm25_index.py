import pickle
import re
from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import config

class BM25Index:
    def __init__(self):
        self.index_path = config.DATA_DIR / "bm25_index.pkl"
        self.bm25 = None
        self.doc_ids = []
        self.load()

    def tokenize(self, text: str) -> List[str]:
        # Simple regex tokenizer + lowercase as requested
        # Split by non-word characters (keeping basic Vietnamese support via unicode props if regex engine supports, 
        # but standard \W might be okay or just split by whitespace and punctuation).
        # Let's use simple alphanumeric split.
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def build(self, chunks: List):
        """
        Builds the BM25 index from a list of Chunk objects.
        chunks: List of Chunk objects (must have .text and .chunk_id attribute)
        """
        print(f"[BM25] Building index for {len(chunks)} chunks...")
        corpus = [self.tokenize(c.text) for c in chunks]
        self.doc_ids = [c.chunk_id for c in chunks]
        
        self.bm25 = BM25Okapi(corpus)
        self.save()
        print(f"[BM25] Index built and saved to {self.index_path}")

    def save(self):
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "doc_ids": self.doc_ids
            }, f)

    def load(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["bm25"]
                    self.doc_ids = data["doc_ids"]
                # print(f"[BM25] Loaded index with {len(self.doc_ids)} docs.")
            except Exception as e:
                print(f"[BM25] Failed to load index: {e}")
                self.bm25 = None
                self.doc_ids = []
        else:
            # print("[BM25] No existing index found.")
            pass

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Returns list of (chunk_id, score)
        """
        if not self.bm25:
            # print("[BM25] Warning: Index not loaded or empty.")
            return []

        tokenized_query = self.tokenize(query)
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip with IDs
        scored_docs = zip(self.doc_ids, scores)
        
        # Sort and top_k
        top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Filter out 0 scores if desired, though BM25 can be negative or 0.
        # Usually we keep strictly positive for search? BM25Okapi in rank_bm25 is usually >= 0 (freq based).
        return top_docs
