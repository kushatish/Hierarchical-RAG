"""Persistent child-chunk search. All corpus/index mutations share a transaction."""
from hashlib import sha256
import json

from core.retrieval import chunk_page, terms

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
 id INTEGER PRIMARY KEY AUTOINCREMENT, config TEXT NOT NULL,
 document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 source TEXT NOT NULL, page INTEGER NOT NULL, parent_id TEXT NOT NULL,
 parent TEXT NOT NULL, text TEXT NOT NULL, terms TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_config ON chunks(config);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(terms, content='chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_insert AFTER INSERT ON chunks BEGIN
 INSERT INTO chunk_search(rowid,terms) VALUES(new.id,new.terms);
END;
CREATE TRIGGER IF NOT EXISTS chunks_delete AFTER DELETE ON chunks BEGIN
 INSERT INTO chunk_search(chunk_search,rowid,terms) VALUES('delete',old.id,old.terms);
END;
CREATE TABLE IF NOT EXISTS indexed_documents (
 config TEXT NOT NULL, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 digest TEXT NOT NULL, PRIMARY KEY(config,document_id)
);
CREATE TABLE IF NOT EXISTS index_state (config TEXT PRIMARY KEY, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS vectors (
 chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
 model TEXT NOT NULL, dimension INTEGER NOT NULL, data BLOB NOT NULL,
 PRIMARY KEY(chunk_id,model)
);
"""

def index_key(settings):
    return sha256(json.dumps(["word-chunks-v1", settings.parent_size, settings.child_size, settings.overlap], separators=(",", ":")).encode()).hexdigest()

def ensure_index(db, settings):
    key = index_key(settings)
    missing = db.execute("""SELECT d.* FROM documents d LEFT JOIN indexed_documents i
        ON i.document_id=d.id AND i.config=? WHERE i.digest IS NULL OR i.digest!=d.digest""", (key,)).fetchall()
    for doc in missing:
        db.execute("DELETE FROM chunks WHERE config=? AND document_id=?", (key, doc["id"]))
        for page, text in enumerate(json.loads(doc["pages"]), 1):
            for parent_no, parent, child in chunk_page(text, settings.parent_size, settings.child_size, settings.overlap):
                db.execute("""INSERT INTO chunks(config,document_id,source,page,parent_id,parent,text,terms)
                    VALUES(?,?,?,?,?,?,?,?)""", (key, doc["id"], doc["name"], page, f'{doc["id"]}:{page}:{parent_no}', parent, child, " ".join(terms(child))))
        db.execute("INSERT OR REPLACE INTO indexed_documents VALUES(?,?,?)", (key, doc["id"], doc["digest"]))
    revision = db.execute("SELECT revision FROM state WHERE id=1").fetchone()[0]
    db.execute("INSERT OR REPLACE INTO index_state VALUES(?,?)", (key, revision))
    return key

def lexical_candidates(db, key, query, limit):
    query_terms = sorted(set(terms(query)))
    if not query_terms:
        return []
    expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in query_terms)
    return [dict(row) for row in db.execute("""SELECT c.*, -bm25(chunk_search) AS score
        FROM chunk_search JOIN chunks c ON c.id=chunk_search.rowid
        WHERE chunk_search MATCH ? AND c.config=?
        ORDER BY bm25(chunk_search),c.id LIMIT ?""", (expression, key, limit))]
