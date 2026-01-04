import argparse
import time
import schedule
from ingest import IngestionPipeline
from rag import RetrievalPipeline
import config
from db import SessionLocal, Chunk
from vector_store import VectorStore
import numpy as np

def run_ingest():
    print(f"[{time.asctime()}] Running Scheduled Ingest...")
    pipeline = IngestionPipeline()
    pipeline.run()

def run_rebuild_index():
    print(f"[{time.asctime()}] Running Scheduled Index Rebuild...")
    # 1. Load active chunks from DB
    session = SessionLocal()
    try:
        active_chunks = session.query(Chunk).filter(Chunk.is_active == True).all()
        if not active_chunks:
            print("No active chunks to rebuild.")
            return

        print(f"Loaded {len(active_chunks)} active chunks from DB.")
        
        # 2. Get embeddings - wait, do we store embeddings in DB? 
        # Standard design: FAISS stores them. If we rebuild, we need the vectors.
        # If we didn't store vectors in DB (blob), we have to regenerate them!
        # Generating embeddings is expensive.
        # Ideally, we should have stored 'embedding' Blob in Chunk table or a separate Vectors table.
        # OR we just rely on the existing FAISS index to 'reconstruct' if using IndexIVF?
        # But IndexFlatIP doesn't support reconstruction automatically unless enabled?
        # Actually IndexFlatIP stores full vectors. We can reconstruct.
        # But wait, if we want to "clean garbage", we only want active ones.
        
        # Current simplified approach: Regeneration (Safe but slow) 
        # OR assume we just stick to Soft Delete filtering in Query for now (Phase 1).
        # But User asked for "Rebuild periodically".
        # Let's implement Regeneration for correctness in this demo context, 
        # knowing that in Prod we'd stash vectors in PGVector/Blob.
        
        db_vectors = []
        db_ids = []
        
        # For simplicity in this demo without vector blob: 
        # We re-embed. It's the only way to be sure without a vector column.
        # (Assuming dataset isn't huge yet).
        
        embedder = IngestionPipeline().embedder
        
        # Batch re-embed
        batch_size = 64
        for i in range(0, len(active_chunks), batch_size):
            batch = active_chunks[i : i+batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            
            vecs = embedder.encode(texts)
            db_vectors.append(vecs)
            db_ids.extend(ids)
            print(f"  Re-embedded batch {i}-{i+len(batch)}")
            
        full_vectors = np.vstack(db_vectors)
        
        # 3. Create new index
        vs = VectorStore()
        vs.rebuild(full_vectors, db_ids)
        print("Vector Index rebuild complete.")
        
        # 4. Rebuild BM25 Index
        from bm25_index import BM25Index
        bm25 = BM25Index()
        bm25.build(active_chunks)

        
    finally:
        session.close()

def run_query(query_text):
    rag = RetrievalPipeline()
    answer, refs = rag.answer_query(query_text)
    print("\n=== CÂU TRẢ LỜI ===")
    print(answer)

def run_scheduler():
    print("Starting Scheduler Mode...")
    # Ingest every X hours
    schedule.every(config.CRAWL_INTERVAL_HOURS).hours.do(run_ingest)
    
    # Rebuild every X days
    schedule.every(config.REBUILD_INTERVAL_DAYS).days.do(run_rebuild_index)
    
    # Run ingest once at startup?
    # run_ingest() 
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def main():
    parser = argparse.ArgumentParser(description="Legal RAG System")
    subparsers = parser.add_subparsers(dest="command")
    
    # Ingest
    subparsers.add_parser("ingest", help="Run ingestion pipeline immediately")
    
    # Rebuild
    subparsers.add_parser("rebuild", help="Force rebuild vector index from active chunks")

    # Query
    q_parser = subparsers.add_parser("query", help="Query the system")
    q_parser.add_argument("text", type=str, help="Question to ask")
    
    # Schedule
    subparsers.add_parser("schedule", help="Run in scheduler mode")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        run_ingest()
    elif args.command == "rebuild":
        run_rebuild_index()
    elif args.command == "query":
        run_query(args.text)
    elif args.command == "schedule":
        run_scheduler()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
