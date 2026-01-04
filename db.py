from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Date, Text, ForeignKey, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, date
import config

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    doc_id = Column(String, primary_key=True) # Canonical ID: issuer_type_number_date
    title = Column(String)
    doc_type = Column(String) # Luat, Nghi dinh, Thong tu...
    issuer = Column(String)
    number_symbol = Column(String)
    issue_date = Column(Date)
    
    effective_from = Column(Date, nullable=True)
    effective_to = Column(Date, nullable=True)
    
    status = Column(String, default=config.STATUS_ACTIVE) # active, inactive
    source_url = Column(String)
    
    # Relationship to get the latest content quickly (manual management required)
    latest_version_id = Column(Integer, nullable=True)

    versions = relationship("DocVersion", back_populates="document")
    chunks = relationship("Chunk", back_populates="document")

class DocVersion(Base):
    __tablename__ = 'doc_versions'

    version_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey('documents.doc_id'))
    
    retrieved_at = Column(DateTime, default=datetime.utcnow)
    content_hash = Column(String) # SHA256 of normalized full text
    raw_file_path = Column(String)
    parser_version = Column(String)

    document = relationship("Document", back_populates="versions")
    chunks = relationship("Chunk", back_populates="version")

class Chunk(Base):
    __tablename__ = 'chunks'

    chunk_id = Column(Integer, primary_key=True, autoincrement=True) # ID for FAISS (SQLite requires Integer for auto-inc)
    version_id = Column(Integer, ForeignKey('doc_versions.version_id'))
    doc_id = Column(String, ForeignKey('documents.doc_id'))
    
    chunk_index = Column(Integer)
    text = Column(Text)
    text_hash = Column(String)
    
    # Metadata for citations and verification
    start_char = Column(Integer)
    end_char = Column(Integer)
    
    # Logic filtering (denormalized from Document for speed if needed, or join)
    is_active = Column(Boolean, default=True)
    valid_from = Column(Date, nullable=True) 
    valid_to = Column(Date, nullable=True)

    document = relationship("Document", back_populates="chunks")
    version = relationship("DocVersion", back_populates="chunks")

# --- DB Setup ---
engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_url_exists(url: str) -> bool:
    db = SessionLocal()
    try:
        return db.query(Document.doc_id).filter(Document.source_url == url).first() is not None
    finally:
        db.close()

def check_urls_exist(urls: list[str]) -> set[str]:
    if not urls:
        return set()
    db = SessionLocal()
    try:
        rows = db.query(Document.source_url).filter(Document.source_url.in_(urls)).all()
        return {r[0] for r in rows}
    finally:
        db.close()
