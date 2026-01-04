import datetime
from sqlalchemy.orm import Session
from db import get_db, Chunk, Document, SessionLocal
from vector_store import VectorStore, TextEmbedder
from text_processor import normalize_text
import config
import openai
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# New imports
from query_planner import QueryPlanner
from bm25_index import BM25Index
from rrf import weighted_rrf
from reranker import Reranker
import numpy as np

@dataclass
class Candidate:
    chunk_id: str
    chunk: Optional[Chunk] = None
    doc: Optional[Document] = None
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    pre_rerank_score: float = 0.0
    final_score: float = 0.0
    # Debug info
    sources: List[str] = field(default_factory=list) # e.g. ["dense_legal", "bm25_facts"]

class RetrievalPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = TextEmbedder()
        self.planner = QueryPlanner()
        self.bm25 = BM25Index()
        self.reranker = Reranker()
        
    def retrieve_candidates(self, plan, valid_at: datetime.date = None) -> List[Candidate]:
        """
        Orchestrates:
        1. Multi-query generation from Plan (Legal Q, Facts, Keywords)
        2. Dense + BM25 Search for each sub-query
        3. RRF Merge
        4. DB Filter (Active/Valid)
        5. Pre-rerank (Aggregation)
        """
        if config.DEBUG_RETRIEVAL:
            print(f"\n[Retrieval] Plan: {plan}")

        # --- 1. Sub-query Generation ---
        sub_queries = []
        # Priority 1: Legal Question
        if plan.legal_question:
            sub_queries.append({"text": plan.legal_question, "type": "legal_question", "weight": 1.2})
        
        # Priority 2: Facts (Cap at 5)
        for f in plan.facts[:5]:
             sub_queries.append({"text": f, "type": "fact", "weight": 1.0})
             
        # Priority 3: Keywords (Cap at 5)
        for k in plan.keywords[:5]:
             sub_queries.append({"text": k, "type": "keyword", "weight": 0.8})
             
        # --- 2. Search (Dense + BM25) ---
        # We collect ranking lists for RRF
        # Each list is [chunk_id, chunk_id, ...] sorted by score
        
        rank_lists = []
        weights = []
        
        # Helper to track scores per candidate for Debug/Pre-rerank
        candidate_map: Dict[str, Candidate] = {}
        
        def get_candidate(cid):
            if cid not in candidate_map:
                candidate_map[cid] = Candidate(chunk_id=cid)
            return candidate_map[cid]

        for q in sub_queries:
            q_text = q['text']
            q_norm = normalize_text(q_text)
            w = q['weight']
            
            # Dense
            q_vec = self.embedder.encode_query(q_norm)
            d_dists, d_ids = self.vector_store.search(q_vec, k=config.TOP_K_DENSE)
            
            # Filter -1 and process
            dense_ids = []
            for cid, score in zip(d_ids, d_dists):
                cid = str(cid) # ensure str
                if cid == "-1": continue
                dense_ids.append(cid)
                
                c = get_candidate(cid)
                c.dense_score = max(c.dense_score, float(score)) # Aggregation: Max
                c.sources.append(f"dense_{q['type']}")
            
            if dense_ids:
                rank_lists.append(dense_ids)
                weights.append(config.WEIGHT_DENSE * w)
                
            # BM25
            bm25_res = self.bm25.search(q_norm, top_k=config.TOP_K_BM25)
            bm25_ids = []
            for cid, score in bm25_res:
                cid = str(cid)
                bm25_ids.append(cid)
                
                c = get_candidate(cid)
                c.bm25_score = max(c.bm25_score, float(score)) # Aggregation: Max
                c.sources.append(f"bm25_{q['type']}")
                
            if bm25_ids:
                rank_lists.append(bm25_ids)
                weights.append(config.WEIGHT_BM25 * w)

        # --- 3. RRF Merge ---
        rrf_results = weighted_rrf(rank_lists, weights, k=config.RRF_K)
        
        # Update RRF scores in candidates
        top_rrf_ids = []
        for cid, rrf_score in rrf_results:
            if cid in candidate_map:
                candidate_map[cid].rrf_score = rrf_score
                top_rrf_ids.append(cid)
                
        # Cut off at TOP_K_RRF (~200)
        top_rrf_ids = top_rrf_ids[:config.TOP_K_RRF]
        
        if config.DEBUG_RETRIEVAL:
            print(f"[Retrieval] Candidates after RRF: {len(top_rrf_ids)}")

        # --- 4. DB Enrichment & Filtering ---
        session = SessionLocal()
        final_candidates = []
        try:
            records = session.query(Chunk, Document).join(Document)\
                .filter(Chunk.chunk_id.in_(top_rrf_ids))\
                .filter(Chunk.is_active == True)\
                .filter(Document.status == config.STATUS_ACTIVE)\
                .all()
                
            # Map back to candidate objects
            record_map = {str(c.chunk_id): (c, d) for c, d in records}
            
            for cid in top_rrf_ids:
                if cid in record_map:
                    chunk, doc = record_map[cid]
                    
                    # Date Logic
                    if valid_at:
                        start = chunk.valid_from or doc.effective_from
                        end = chunk.valid_to or doc.effective_to
                        if start and start > valid_at: continue
                        if end and end < valid_at: continue
                    
                    # Passed filters
                    cand = candidate_map[cid]
                    cand.chunk = chunk
                    cand.doc = doc
                    final_candidates.append(cand)
                    
        finally:
            session.close()

        # --- 5. Pre-Rerank (Lightweight Aggregation) ---
        # pre_score = Score(RRF) + norm(Dense) + norm(BM25)
        # For simplicity in this demo, we just use RRF score primarily, 
        # or we can implement the weighted sum as requested in Phase D.
        
        # Let's normalize dense and bm25 scores roughly (just for consistent magnitude)
        # Actually RRF alone is very strong. Let's just trust RRF for Pre-rank order.
        # But User asked for D1: a*rrf + b*dense + c*bm25
        
        # Normalize in memory
        max_dense = max((c.dense_score for c in final_candidates), default=1.0) or 1.0
        max_bm25 = max((c.bm25_score for c in final_candidates), default=1.0) or 1.0
        
        for c in final_candidates:
            norm_dense = c.dense_score / max_dense
            norm_bm25 = c.bm25_score / max_bm25
            
            c.pre_rerank_score = (
                config.PRE_RERANK_ALPHA * c.rrf_score +
                config.PRE_RERANK_BETA * norm_dense +
                config.PRE_RERANK_GAMMA * norm_bm25
            )
            
        final_candidates.sort(key=lambda x: x.pre_rerank_score, reverse=True)
        return final_candidates[:config.TOP_K_PRE_RERANK]

    def build_context(self, candidates: List[Candidate]):
        context = ""
        citations = []
        seen_docs = set()
        
        for i, item in enumerate(candidates):
            c = item.chunk
            d = item.doc
            
            # Improved Citation source ref
            source_ref = f"[{d.doc_type} {d.number_symbol}]"
            context += f"Source {i+1} {source_ref}: {c.text}\n\n"
            
            citation_entry = f"{d.doc_type} {d.number_symbol} ({d.issue_date}) - {d.source_url}"
            if explanation_match := self._extract_provision(c.text):
                 citation_entry += f" [{explanation_match}]"
                 
            if d.doc_id not in seen_docs:
                citations.append(citation_entry)
                seen_docs.add(d.doc_id)
                
        return context, citations

    def _extract_provision(self, text):
        import re
        match = re.search(r'(Điều \d+)', text)
        return match.group(1) if match else ""

    def answer_query(self, query: str):
        if config.DEBUG_RETRIEVAL:
            print(f"\n=== PROCESSING QUERY: {query} ===")

        # 1. Query Planning
        plan = self.planner.plan(query)
        
        # 2. Retrieve & Pre-rank
        candidates = self.retrieve_candidates(plan, valid_at=datetime.date.today())
        
        if not candidates:
            return "Xin lỗi, không tìm thấy dữ liệu phù hợp trong cơ sở dữ liệu hiện hành.", []
            
        # 3. Cross-Encoder Rerank (Deep)
        # Input format for reranker: list of (id, text)
        # Reranker takes original query vs candidate text. 
        # Or better: (Plan.legal_question, chunk.text) as it captures intent better than long query.
        
        rerank_input = [(c.chunk_id, c.chunk.text) for c in candidates]
        
        # We rerank based on Legal Question (intent) for better precision
        # Alternatively, use Full Query if it's not too long.
        rerank_query = plan.legal_question if plan.legal_question else query
        
        scored_indices = self.reranker.rerank(rerank_query, rerank_input, top_k=config.TOP_K_FINAL)
        
        best_candidates = []
        for idx, score in scored_indices:
            cand = candidates[idx]
            cand.final_score = score
            best_candidates.append(cand)
        
        if config.DEBUG_RETRIEVAL:
            print("\n[Top Reranked Results]")
            for i, c in enumerate(best_candidates):
                print(f"{i+1}. [{c.final_score:.4f}] {c.doc.doc_type} {c.doc.number_symbol} - {c.chunk.text[:50]}...")

        # 4. Context
        context, refs = self.build_context(best_candidates)
        
        # 5. Prompt
        prompt = f"""Bạn là trợ lý pháp lý AI. Dựa vào thông tin sau để trả lời câu hỏi.
        YÊU CẦU AN TOÀN:
        - Chỉ dùng thông tin trong Context được cung cấp bên dưới.
        - Nếu thông tin không đủ để trả lời, HÃY NÓI "Không đủ thông tin trong văn bản được cung cấp".
        - TUYỆT ĐỐI KHÔNG tự bịa ra điều luật hay suy đoán ngoài văn bản.
        - Mỗi ý trả lời phải gắn kèm trích dẫn (VD: [Nghị định 123...]).
        
        Context:
        {context}
        
        Câu hỏi: {query}
        Trả lời:"""
        
        if config.OPENAI_API_KEY:
            try:
                client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"Lỗi gọi LLM: {e}"
        else:
             answer = "Lỗi: Chưa cấu hình OPENAI_API_KEY."
            
        return answer, refs

