from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.ingest import Document, DATABASE_URL

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

for doc in session.query(Document).order_by(Document.id.desc()).limit(10):
    print(doc.id, doc.source, doc.chunk_index)
    print("preview:", (doc.content[:200] + "...") if len(doc.content) > 200 else doc.content)
    print("embedding length:", len(doc.embedding) if doc.embedding is not None else None)
    print("embedding sample:", doc.embedding[:8] if doc.embedding else None)
    print("-" * 40)

session.close()