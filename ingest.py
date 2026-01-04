import os
import hashlib
import datetime
import shutil
import feedparser
from sqlalchemy.orm import Session

from db import get_db, Document, DocVersion, Chunk, init_db, check_url_exists, check_urls_exist
from extractors import extract_text_from_file
from text_processor import normalize_text, smart_legal_chunking
from vector_store import VectorStore, TextEmbedder
import config
from slugify import slugify
from sources import vbpl_crawler
from unidecode import unidecode
import re
import numpy as np
import signal
import sys
import time

class IngestionPipeline:
    def __init__(self):
        # Initialize resources

        init_db()
        self.vector_store = VectorStore()
        self.embedder = TextEmbedder()
        self.must_stop = False
        self._seen_content_hashes = set() # Runtime duplicate guard
        
        # Batch Save Config
        self.docs_since_save = 0
        self.last_save_ts = time.time()
        self.SAVE_EVERY_DOCS = int(os.getenv("FAISS_SAVE_EVERY_DOCS", "25"))
        self.SAVE_EVERY_SEC  = int(os.getenv("FAISS_SAVE_EVERY_SEC", "300"))
        
        # Signal Handling
        signal.signal(signal.SIGINT, self.handle_sigint)
        signal.signal(signal.SIGTERM, self.handle_sigint)

    def handle_sigint(self, signum, frame):
        print("\n\n!!! RECEIVED STOP SIGNAL (Ctrl+C). Gracefully stopping... !!!")
        self.must_stop = True
        
    def _maybe_save_faiss(self, force=False):
        now = time.time()
        if force or self.docs_since_save >= self.SAVE_EVERY_DOCS or (now - self.last_save_ts) >= self.SAVE_EVERY_SEC:
            if self.docs_since_save > 0 or force:
                print(f"  [Auto-Save] Saving FAISS index (docs={self.docs_since_save})...")
                self.vector_store.save()
            self.docs_since_save = 0
            self.last_save_ts = now

    def record_failed(self, url: str, reason: str):
        os.makedirs("data", exist_ok=True)
        with open("data/failed_urls.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.utcnow().isoformat()}\t{url}\t{reason}\n")

    def discover_sources(self):
        """
        1. Retry Failed
        2. Bootstrap VBPL
        3. RSS Updates
        """
        

        # Helper Inner Functions
        def parse_date_any(s):
            if not s: return None
            s = str(s).strip()
            if m := re.match(r"(\d{4})-(\d{2})-(\d{2})", s): return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if m := re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s): return datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            return None

        def extract_meta_light(html_text: str):
            text = html_text
            num = m.group(1).strip() if (m := re.search(r"Số\s*:\s*([0-9A-Za-z\/\.\-_]+)", text, re.I)) else None
            
            issue = None
            if m := re.search(r"Ban\s*hành\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.I): issue = parse_date_any(m.group(1))
            if not issue and (m := re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", text, re.I)):
                issue = datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))

            eff = parse_date_any(m.group(1)) if (m := re.search(r"Ngày\s*có\s*hiệu\s*lực\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.I)) else None
            title = m.group(1).strip() if (m := re.search(r"Trích\s*yếu\s*:\s*([^\n\r<]+)", text, re.I)) else None
            return num, issue, eff, title

        def check_known_callback(urls: list[str]) -> tuple[bool, float]:
            if not urls: return False, 0.0
            known = check_urls_exist(urls)
            return False, len(known) / len(urls)
            
        def resolve_attachment(entry_url, html):
            """
            Find attachment (PDF/DOC) in RSS HTML content.
            """
            try:
                from bs4 import BeautifulSoup
                from urllib.parse import urljoin
                soup = BeautifulSoup(html, "lxml")
                # Priority 1: Direct file link
                for a in soup.find_all("a", href=True):
                    href = a["href"].lower()
                    if href.endswith(".pdf") or href.endswith(".doc") or href.endswith(".docx"):
                        return urljoin(entry_url, a["href"])
                # Priority 2: "Tải về" link
                for a in soup.find_all("a", href=True):
                    if "tải về" in a.get_text(" ", strip=True).lower():
                        return urljoin(entry_url, a["href"])
            except Exception:
                pass
            return None

        # --- 0. Retry Failed URLs ---
        failed_path = "data/failed_urls_vn.txt"
        if os.path.exists(failed_path):
            print("--- Mode: Retry Failed URLs ---")
            try:
                with open(failed_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # Simple retry logic: process simple batch
                for line in lines[-200:]: # Last 200
                    if self.must_stop: break
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        url = parts[1]
                        if check_url_exists(url): continue
                        
                        try:
                            fp = vbpl_crawler.download_html(url, os.path.join(config.RAW_DATA_DIR, "retry_html"))
                            yield {
                                "title": "Retry Document",
                                "doc_type": "Retried",
                                "issuer": "Unknown",
                                "number_symbol": "RETRY_" + hashlib.md5(url.encode()).hexdigest(),
                                "issue_date": datetime.date.today(),
                                "effective_from": None,
                                "source_url": url,
                                "file_path": fp
                            }
                        except Exception as e:
                            print(f"Retry failed for {url}: {e}")
            except Exception as e:
                print(f"Error reading failed_urls: {e}")

        # --- 1. VBPL Bootstrap ---
        print("--- Mode: Bootstrapping from VBPL Seed Lists ---")
        save_dir = os.path.join(config.RAW_DATA_DIR, "vbpl_html")
        os.makedirs(save_dir, exist_ok=True)

        for seed in vbpl_crawler.SEED_LISTS:
            if self.must_stop: break
            
            list_url = seed["list_url"]
            doc_type = seed["doc_type"]
            issuer_default = seed["issuer_default"]
            seed_key = f"{doc_type}|{list_url}"

            iterator = vbpl_crawler.iter_toanvan_urls(list_url, seed_key=seed_key, check_known_func=check_known_callback)
            
            for page, offset, toanvan_url in iterator:
                if self.must_stop: break

                if check_url_exists(toanvan_url):
                    vbpl_crawler.save_checkpoint(seed_key, page, offset + 1)
                    continue

                try:
                    file_path = vbpl_crawler.download_html(toanvan_url, save_dir)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        html = f.read()

                    num, issue_date, eff_from, title_hint = extract_meta_light(html)
                    number_symbol = num or f"VBPL_{os.path.splitext(os.path.basename(file_path))[0]}"
                    safe_issue_date = issue_date or datetime.date(1970, 1, 1) # Default
                    title = title_hint or f"{doc_type} {number_symbol}"
                    
                    item_payload = {
                        "title": title,
                        "doc_type": doc_type,
                        "issuer": issuer_default,
                        "number_symbol": number_symbol,
                        "issue_date": safe_issue_date,
                        "effective_from": eff_from,
                        "source_url": toanvan_url,
                        "file_path": file_path,
                        # Attach Checkpoint info to pass to 'run' loop
                        "_ckpt": {"seed_key": seed_key, "page": page, "offset": offset + 1} 
                    }
                    
                    yield item_payload
                    
                except Exception as e:
                    print(f"Error processing {toanvan_url}: {e}")
                    self.record_failed(toanvan_url, str(e))
                    vbpl_crawler.save_checkpoint(seed_key, page, offset + 1)

        # --- 2. RSS Incremental Update ---
        if not self.must_stop:
            print("--- Mode: RSS Check (Cong Bao) ---")
            rss_urls = [
                "http://congbao.chinhphu.vn/cac_van_ban_moi_ban_hanh.rss",
                "http://congbao.chinhphu.vn/cac_so_cong_bao_moi_dang.rss",
            ]

            for rss_url in rss_urls:
                if self.must_stop: break
                try:
                    feed = feedparser.parse(rss_url)
                    if getattr(feed, "bozo", 0):
                        print(f"[WARN] RSS parse error: {getattr(feed,'bozo_exception',None)}")

                    print(f"RSS {rss_url} entries: {len(feed.entries)}")
                    for e in feed.entries:
                        if self.must_stop: break
                        
                        link = getattr(e, "link", None)
                        if not link or check_url_exists(link):
                            continue
                        
                        title = getattr(e, "title", "") or "Công báo"
                        
                        # Use download to temp buf first to check redirects/content
                        try:
                            # 1. Download Landing HTML
                            fp_html = vbpl_crawler.download_html(link, os.path.join(config.RAW_DATA_DIR, "congbao_html"))
                            with open(fp_html, "r", encoding="utf-8", errors="ignore") as f:
                                html_landing = f.read()
                            
                            # 2. Resolve PDF/DOC attachment
                            attachment_url = resolve_attachment(link, html_landing)
                            
                            final_fp = fp_html
                            item_url = link
                            
                            if attachment_url:
                                print(f"  [RSS] Resolved attachment: {attachment_url}")
                                try:
                                    final_fp = vbpl_crawler.download_html(attachment_url, os.path.join(config.RAW_DATA_DIR, "congbao_files"))
                                    # We use the attachment URL as the source_url for uniqueness if we only care about file?
                                    # Or keep the RSS link as source_url (for linking back)?
                                    # Usually keep the RSS link 'link' as source_url to match 'check_url_exists'
                                    # BUT we process the file content.
                                except Exception as att_e:
                                    print(f"  [WARN] Failed to download attachment {attachment_url}: {att_e}. Fallback to HTML.")
                            
                            yield {
                                "title": title,
                                "doc_type": "Công báo",
                                "issuer": "Chính phủ",
                                "number_symbol": title[:50],
                                "issue_date": datetime.date.today(),
                                "effective_from": None,
                                "source_url": item_url,
                                "file_path": final_fp,
                            }
                        except Exception as dl_e:
                            print(f"RSS item fetch failed {link}: {dl_e}")
                            self.record_failed(link, str(dl_e))

                except Exception as e:
                     print(f"RSS fetch failed {rss_url}: {e}")

    def generate_canonical_id(self, item):
        import re
        d = item['issue_date'] if item['issue_date'] else "nodate"
        raw = f"{item['issuer']}_{item['doc_type']}_{item['number_symbol']}_{d}"
        raw = unidecode(raw).lower()
        raw = raw.replace("/", "-").replace(" ", "-")
        raw = re.sub(r"[^a-z0-9\-_]+", "", raw)
        return raw[:240]

    def calculate_hash(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def process_document(self, db: Session, item: dict):
        doc_id = self.generate_canonical_id(item)
        print(f"Processing {doc_id}...")
        
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if not doc:
            doc = Document(
                doc_id=doc_id,
                title=item['title'],
                doc_type=item['doc_type'],
                issuer=item['issuer'],
                number_symbol=item['number_symbol'],
                issue_date=item['issue_date'],
                effective_from=item.get('effective_from'),
                source_url=item['source_url'],
                status=config.STATUS_ACTIVE
            )
            db.add(doc)
            db.flush()
            print("  New Document created.")
        
        fpath = item.get('file_path')
        if not os.path.exists(fpath):
            print(f"  File not found: {fpath}. Skipping.")
            return

        text, used_ocr = extract_text_from_file(fpath)
        print(f"  Extracted chars: {len(text)} | used_ocr={used_ocr}")
        

        normalized_text = normalize_text(text)
        print(f"  Normalized chars: {len(normalized_text)}")
        content_hash = self.calculate_hash(normalized_text)
        
        # --- Runtime Duplicate Guard (Requirement D) ---
        if content_hash in self._seen_content_hashes:
            print("  [WARN] Duplicate content hash found in this run. Skipping.")
            return
        self._seen_content_hashes.add(content_hash)
        # -----------------------------------------------
        
        latest_ver = None
        if doc.latest_version_id:
             latest_ver = db.query(DocVersion).filter(DocVersion.version_id == doc.latest_version_id).first()
             
        if latest_ver and latest_ver.content_hash == content_hash:
            print("  Content unchanged. Skipping.")
            return
            
        print("  New version detected. Ingesting...")
        
        # --- Consistency Logic: Get Old IDs First (Requirement 3.7) ---
        old_ids = []
        if latest_ver:
            old_chunks = db.query(Chunk.chunk_id).filter(Chunk.version_id == latest_ver.version_id).all()
            old_ids = [c[0] for c in old_chunks]
        # ----------------------------------------------------------------
        
        new_version = DocVersion(
            doc_id=doc_id,
            content_hash=content_hash,
            raw_file_path=fpath,
            parser_version="v3.1-ref" 
        )
        db.add(new_version)
        db.flush()
        
        doc.latest_version_id = new_version.version_id
        
        chunks_text = smart_legal_chunking(normalized_text)
        print(f"  Split into {len(chunks_text)} chunks.")
        
        # Batch DB Insert (Requirement 3.6)
        chunk_rows = []
        for idx, chunk_txt in enumerate(chunks_text):
            c = Chunk(
                version_id=new_version.version_id,
                doc_id=doc_id,
                chunk_index=idx,
                text=chunk_txt,
                text_hash=self.calculate_hash(chunk_txt),
                is_active=True,
                valid_from=doc.effective_from,
                valid_to=doc.effective_to,
                start_char=0,
                end_char=0
            )
            chunk_rows.append(c)
            
        if chunk_rows:
            db.add_all(chunk_rows)
            db.flush() # Assigns IDs
            
            ids = [c.chunk_id for c in chunk_rows]
            vectors_np = self.embedder.encode(chunks_text)
            self.vector_store.add_vectors(vectors_np, ids)
            
        if latest_ver:
            # Mark inactive
            db.query(Chunk).filter(Chunk.version_id == latest_ver.version_id).update({"is_active": False})

        db.commit() # Commit DB first
        
        # Remove old IDs from FAISS only after DB commit success (Requirement 3.7)
        if old_ids:
            print(f"  Removing {len(old_ids)} old chunks from FAISS.")
            self.vector_store.remove_ids(old_ids)

        self.docs_since_save += 1
        print("  Ingestion complete.")

    def run(self):
        print("Starting Ingestion Pipeline...")
        db_gen = get_db()
        session = next(db_gen)
        try:
            for item in self.discover_sources():
                if self.must_stop:
                    print("Stop signal detected. Exiting loop.")
                    break
                
                ck = item.get("_ckpt")
                try:
                    self.process_document(session, item)
                    
                    # Ack checkpoint AFTER success
                    if ck:
                        vbpl_crawler.save_checkpoint(ck["seed_key"], ck["page"], ck["offset"])
                        
                    self._maybe_save_faiss()
                    
                except Exception as e:
                    print(f"Error processing {item.get('title')}: {e}")
                    session.rollback()
                    url = item.get("source_url", "")
                    self.record_failed(url, repr(e))
                    
                    # Still Ack checkpoint to avoid loop
                    if ck:
                        vbpl_crawler.save_checkpoint(ck["seed_key"], ck["page"], ck["offset"])
        finally:
            print("Finalizing graceful stop...")
            self._maybe_save_faiss(force=True)
            session.commit() # Ensure DB state
            session.close()
            print("Pipeline Stopped.")
    
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
