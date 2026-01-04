# inspect.py
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, inspect as sa_inspect
from sqlalchemy.orm import Session

import config
from db import SessionLocal, Document, DocVersion, Chunk


def _preview(s: Optional[str], n: int = 500) -> str:
    s = s or ""
    s = s.replace("\r", " ").replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s


def _print_kv(title: str, rows: List[Tuple[Any, ...]], headers: List[str]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(v)))

    fmt = "  " + " | ".join("{:" + str(w) + "}" for w in col_widths)
    print(fmt.format(*headers))
    print("  " + "-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt.format(*[str(v) for v in r]))


def cmd_schema(db: Session) -> None:
    print(f"DB_PATH: {config.DB_PATH}")
    p = Path(config.DB_PATH)
    print(f"DB exists: {p.exists()} | size={p.stat().st_size if p.exists() else 'N/A'} bytes")

    insp = sa_inspect(db.bind)
    tables = insp.get_table_names()
    print("\nTables:", tables)

    for t in tables:
        print(f"\n--- Table: {t} ---")
        cols = insp.get_columns(t)
        for c in cols:
            # name, type, nullable, default
            print(f"  {c['name']:16s} {str(c['type']):18s} nullable={c.get('nullable')} default={c.get('default')}")


def cmd_summary(db: Session) -> None:
    n_docs = db.query(func.count(Document.doc_id)).scalar() or 0
    n_versions = db.query(func.count(DocVersion.version_id)).scalar() or 0
    n_chunks = db.query(func.count(Chunk.chunk_id)).scalar() or 0
    n_active_chunks = db.query(func.count(Chunk.chunk_id)).filter(Chunk.is_active == True).scalar() or 0

    print("=== SUMMARY ===")
    print(f"Documents     : {n_docs}")
    print(f"DocVersions   : {n_versions}")
    print(f"Chunks        : {n_chunks}")
    print(f"Active Chunks : {n_active_chunks}")

    # group by doc_type
    rows = (
        db.query(Document.doc_type, func.count(Document.doc_id))
        .group_by(Document.doc_type)
        .order_by(func.count(Document.doc_id).desc())
        .all()
    )
    _print_kv("Documents by doc_type", rows, ["doc_type", "count"])

    # group by status
    rows2 = (
        db.query(Document.status, func.count(Document.doc_id))
        .group_by(Document.status)
        .order_by(func.count(Document.doc_id).desc())
        .all()
    )
    _print_kv("Documents by status", rows2, ["status", "count"])


def cmd_latest_docs(db: Session, n: int = 10) -> None:
    # Document has no created_at; use latest_version_id to approximate "newness"
    rows = (
        db.query(Document)
        .order_by(Document.latest_version_id.desc().nullslast(), Document.doc_id.desc())
        .limit(n)
        .all()
    )

    print(f"\n=== LATEST {n} DOCUMENTS (approx) ===")
    for d in rows:
        print("\n---")
        print(f"doc_id          : {d.doc_id}")
        print(f"title           : {d.title}")
        print(f"doc_type        : {d.doc_type}")
        print(f"issuer          : {d.issuer}")
        print(f"number_symbol   : {d.number_symbol}")
        print(f"issue_date      : {d.issue_date}")
        print(f"effective_from  : {d.effective_from}")
        print(f"effective_to    : {d.effective_to}")
        print(f"status          : {d.status}")
        print(f"source_url      : {d.source_url}")
        print(f"latest_version  : {d.latest_version_id}")


def cmd_latest_chunks(db: Session, n: int = 10, only_active: bool = False) -> None:
    q = db.query(Chunk).order_by(Chunk.chunk_id.desc())
    if only_active:
        q = q.filter(Chunk.is_active == True)
    rows = q.limit(n).all()

    print(f"\n=== LATEST {n} CHUNKS{' (active only)' if only_active else ''} ===")
    for c in rows:
        print("\n---")
        print(f"chunk_id     : {c.chunk_id}")
        print(f"doc_id       : {c.doc_id}")
        print(f"version_id   : {c.version_id}")
        print(f"chunk_index  : {c.chunk_index}")
        print(f"is_active    : {c.is_active}")
        print(f"valid_from   : {c.valid_from}")
        print(f"valid_to     : {c.valid_to}")
        print(f"text_len     : {len(c.text or '')}")
        print(f"text_hash    : {c.text_hash}")
        print(f"preview      : {_preview(c.text, 800)}")


def cmd_doc(db: Session, doc_id: str, show_chunks: bool, max_chunks: int) -> None:
    d = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not d:
        print(f"Not found doc_id={doc_id}")
        return

    print("\n=== DOCUMENT ===")
    print(f"doc_id          : {d.doc_id}")
    print(f"title           : {d.title}")
    print(f"doc_type        : {d.doc_type}")
    print(f"issuer          : {d.issuer}")
    print(f"number_symbol   : {d.number_symbol}")
    print(f"issue_date      : {d.issue_date}")
    print(f"effective_from  : {d.effective_from}")
    print(f"effective_to    : {d.effective_to}")
    print(f"status          : {d.status}")
    print(f"source_url      : {d.source_url}")
    print(f"latest_version  : {d.latest_version_id}")

    vers = (
        db.query(DocVersion)
        .filter(DocVersion.doc_id == doc_id)
        .order_by(DocVersion.version_id.desc())
        .all()
    )
    print(f"\nVersions: {len(vers)}")
    for v in vers[:10]:
        chunk_cnt = db.query(func.count(Chunk.chunk_id)).filter(Chunk.version_id == v.version_id).scalar() or 0
        raw_exists = Path(v.raw_file_path).exists() if v.raw_file_path else False
        print("\n--- version ---")
        print(f"version_id    : {v.version_id}")
        print(f"retrieved_at  : {v.retrieved_at}")
        print(f"parser_version: {v.parser_version}")
        print(f"content_hash  : {v.content_hash}")
        print(f"raw_file_path : {v.raw_file_path} (exists={raw_exists})")
        print(f"chunks        : {chunk_cnt}")

    if show_chunks and d.latest_version_id:
        print(f"\n=== CHUNKS of latest_version_id={d.latest_version_id} (max {max_chunks}) ===")
        chunks = (
            db.query(Chunk)
            .filter(Chunk.version_id == d.latest_version_id)
            .order_by(Chunk.chunk_index.asc())
            .limit(max_chunks)
            .all()
        )
        for c in chunks:
            print("\n---")
            print(f"chunk_id    : {c.chunk_id} | idx={c.chunk_index} | active={c.is_active} | len={len(c.text or '')}")
            print(_preview(c.text, 1200))


def cmd_search(db: Session, q: str, n: int = 10, only_active: bool = False) -> None:
    query = db.query(Chunk).filter(Chunk.text != None).filter(Chunk.text.contains(q))
    if only_active:
        query = query.filter(Chunk.is_active == True)
    rows = query.order_by(Chunk.chunk_id.desc()).limit(n).all()

    print(f"\n=== SEARCH chunks contains {q!r} (top {n}) ===")
    for c in rows:
        print("\n---")
        print(f"chunk_id   : {c.chunk_id} | doc_id={c.doc_id} | version_id={c.version_id} | idx={c.chunk_index} | active={c.is_active}")
        print(_preview(c.text, 1200))


def cmd_audit(db: Session, out: Optional[str], sample: int = 50) -> None:
    """
    Audit nhanh để phát hiện:
    - chunk quá ngắn
    - chunk có dấu hiệu HTML/script
    - doc issue_date NULL
    - content_hash trùng nhiều (dấu hiệu crawl bị chặn và lấy cùng 1 trang)
    """
    report: Dict[str, Any] = {}

    # Short chunks
    short_rows = (
        db.query(Chunk.chunk_id, Chunk.doc_id, func.length(Chunk.text))
        .filter(Chunk.text != None)
        .order_by(func.length(Chunk.text).asc())
        .limit(sample)
        .all()
    )
    report["short_chunks"] = [
        {"chunk_id": int(cid), "doc_id": did, "len": int(l or 0)} for cid, did, l in short_rows
    ]

    # Suspicious HTML chunks
    suspicious = (
        db.query(Chunk)
        .filter(Chunk.text != None)
        .filter(
            (Chunk.text.ilike("%<html%")) |
            (Chunk.text.ilike("%<script%")) |
            (Chunk.text.ilike("%<!doctype%")) |
            (Chunk.text.ilike("%var %"))  # crude JS hint
        )
        .order_by(Chunk.chunk_id.desc())
        .limit(sample)
        .all()
    )
    report["suspicious_html_chunks"] = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "version_id": c.version_id,
            "len": len(c.text or ""),
            "preview": _preview(c.text, 300)
        }
        for c in suspicious
    ]

    # Missing dates
    missing_dates = (
        db.query(Document.doc_id, Document.doc_type, Document.number_symbol, Document.title, Document.source_url)
        .filter(Document.issue_date == None)
        .limit(sample)
        .all()
    )
    report["documents_missing_issue_date"] = [
        {"doc_id": r[0], "doc_type": r[1], "number_symbol": r[2], "title": r[3], "source_url": r[4]}
        for r in missing_dates
    ]

    # Duplicate content_hash (top duplicates)
    dup_hash = (
        db.query(DocVersion.content_hash, func.count(DocVersion.version_id).label("cnt"))
        .filter(DocVersion.content_hash != None)
        .group_by(DocVersion.content_hash)
        .having(func.count(DocVersion.version_id) > 1)
        .order_by(func.count(DocVersion.version_id).desc())
        .limit(20)
        .all()
    )
    report["duplicate_content_hash_top"] = [{"content_hash": h, "count": int(c)} for h, c in dup_hash]

    # For each top dup hash, list a few examples
    examples = []
    for h, _ in dup_hash[:5]:
        vers = (
            db.query(DocVersion.version_id, DocVersion.doc_id, DocVersion.raw_file_path, DocVersion.retrieved_at)
            .filter(DocVersion.content_hash == h)
            .order_by(DocVersion.version_id.desc())
            .limit(5)
            .all()
        )
        examples.append({
            "content_hash": h,
            "examples": [
                {"version_id": int(v[0]), "doc_id": v[1], "raw_file_path": v[2], "retrieved_at": str(v[3])}
                for v in vers
            ]
        })
    report["duplicate_content_hash_examples"] = examples

    # Basic counts
    report["counts"] = {
        "documents": int(db.query(func.count(Document.doc_id)).scalar() or 0),
        "versions": int(db.query(func.count(DocVersion.version_id)).scalar() or 0),
        "chunks": int(db.query(func.count(Chunk.chunk_id)).scalar() or 0),
        "active_chunks": int(db.query(func.count(Chunk.chunk_id)).filter(Chunk.is_active == True).scalar() or 0),
    }

    print("\n=== AUDIT REPORT (summary) ===")
    print(json.dumps({
        "counts": report["counts"],
        "short_chunks": report["short_chunks"][:10],
        "suspicious_html_chunks": report["suspicious_html_chunks"][:5],
        "documents_missing_issue_date": report["documents_missing_issue_date"][:5],
        "duplicate_content_hash_top": report["duplicate_content_hash_top"][:10],
    }, ensure_ascii=False, indent=2))

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved full audit report to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Inspect SQLite (documents / doc_versions / chunks)")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("schema", help="Show tables + columns")
    sub.add_parser("summary", help="Show high-level counts and distributions")

    p_docs = sub.add_parser("latest-docs", help="Show latest documents (approx)")
    p_docs.add_argument("--n", type=int, default=10)

    p_chunks = sub.add_parser("latest-chunks", help="Show latest chunks")
    p_chunks.add_argument("--n", type=int, default=10)
    p_chunks.add_argument("--active", action="store_true", help="Only active chunks")

    p_doc = sub.add_parser("doc", help="Show one document + its versions (+ chunks optional)")
    p_doc.add_argument("doc_id", type=str)
    p_doc.add_argument("--chunks", action="store_true", help="Also show chunks of latest version")
    p_doc.add_argument("--max-chunks", type=int, default=10)

    p_search = sub.add_parser("search", help="Search keyword in chunk text")
    p_search.add_argument("--q", required=True)
    p_search.add_argument("--n", type=int, default=10)
    p_search.add_argument("--active", action="store_true", help="Only active chunks")

    p_audit = sub.add_parser("audit", help="Audit data quality quickly")
    p_audit.add_argument("--out", type=str, default="", help="Write full report json to this path")
    p_audit.add_argument("--sample", type=int, default=50)

    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.cmd == "schema":
            cmd_schema(db)
        elif args.cmd == "summary" or args.cmd is None:
            cmd_summary(db)
        elif args.cmd == "latest-docs":
            cmd_latest_docs(db, n=args.n)
        elif args.cmd == "latest-chunks":
            cmd_latest_chunks(db, n=args.n, only_active=args.active)
        elif args.cmd == "doc":
            cmd_doc(db, doc_id=args.doc_id, show_chunks=args.chunks, max_chunks=args.max_chunks)
        elif args.cmd == "search":
            cmd_search(db, q=args.q, n=args.n, only_active=args.active)
        elif args.cmd == "audit":
            cmd_audit(db, out=args.out or None, sample=args.sample)
        else:
            parser.print_help()
    finally:
        db.close()


if __name__ == "__main__":
    main()
