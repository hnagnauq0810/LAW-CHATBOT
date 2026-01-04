from typing import List, Tuple
from sentence_transformers import CrossEncoder
import config
import torch

class Reranker:
    def __init__(self):
        # Using a lightweight but effective cross-encoder
        # 'cross-encoder/ms-marco-MiniLM-L-6-v2' is fast and good for English.
        # For MULTILINGUAL (Vietnamese), we should use 'cross-encoder/mmarco-mMiniLM-v2-L12-H384-v1' or similar.
        # Assuming we stick to something reasonable. Let's start with mMiniLM if available, or just standard for demo.
        # Given "ZZZDemo" context and Vietnamese, let's look for a multilingual one.
        # "amberoad/bert-multilingual-passage-reranking-msmarco" is an option but old.
        # "cross-encoder/ms-marco-MiniLM-L-6-v2" is strictly English but sometimes works okayish on keywords.
        # LET'S USE 'cross-encoder/ms-marco-MiniLM-L-6-v2' as placeholder or 'impira/layoutlm-invoices'?? No.
        # Let's use 'sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens' -> No that's bi-encoder.
        # We will use 'cross-encoder/ms-marco-MiniLM-L-6-v2' as requested in plan example, 
        # but acknowledging it might need a Vietnamese specific model later (like BGE-M3 reranker).
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
        self.device = config.DEVICE
        self.model = None

    def _load_model(self):
        if self.model is None:
            # Lazy load
            try:
                print(f"[Reranker] Loading model {self.model_name}...")
                self.model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                print(f"[Reranker] Failed to load model: {e}")

    def rerank(self, query: str, candidates: List[Tuple[str, str]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Reranks a list of (id, text) pairs.
        Returns top_k (original_index_in_candidates, score)
        """
        self._load_model()
        if not self.model or not candidates:
             # Return as is with dummy scores if model fails
             return [(i, 0.0) for i in range(len(candidates[:top_k]))]

        # Prepare pairs: (query, text)
        pairs = [[query, text] for _, text in candidates]
        
        # Batch prediction
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        
        # Attach original index to sort and return
        # [(index, score), ...]
        scored_results = []
        for i, score in enumerate(scores):
            scored_results.append((i, float(score)))
            
        # Sort descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results[:top_k]
