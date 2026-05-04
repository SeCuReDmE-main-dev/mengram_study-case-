"""
SQLiteVecVectorStore — Phase 2 backend using sqlite-vec extension.

Leverages the sqlite-vec extension for ANN vector search in SQLite.
"""

import sqlite3
import numpy as np
from typing import Optional, List

import sqlite_vec

from engine.vector.base import BaseVectorStore, SearchResult


class SQLiteVecVectorStore(BaseVectorStore):
    """
    SQLite-based vector storage with sqlite-vec extension for fast ANN search.
    
    Uses the sqlite-vec extension which provides:
    - vec0() virtual table for vector storage
    - vec_distance_cosine() for cosine distance calculations
    - MATCH operator for approximate nearest-neighbor search
    
    Suitable for vaults of 10K+ notes with maintained accuracy.
    """

    def __init__(self, db_path: str = ":memory:", embedder=None,
                 dimension: int = 384):
        super().__init__(dimension=dimension, embedder=embedder)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._enable_vec_extension()
        self._create_tables()

    def _enable_vec_extension(self):
        """Enable the sqlite-vec extension using platform-safe loader."""
        try:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sqlite-vec extension: {e}. "
                f"Ensure sqlite-vec is properly installed (pip install sqlite-vec)."
            ) from e

    def _create_tables(self):
        # Build vec0 schema dynamically from configured dimension
        vec_dim_sql = f"embedding float[{self.dimension}]"
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                position INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
                {vec_dim_sql}
            );
            
            CREATE TABLE IF NOT EXISTS vec_map (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_chunks_entity ON chunks(entity_id);
        """)
        self.conn.commit()

    def _serialize_vec(self, vector: np.ndarray) -> bytes:
        """Validate, normalize, and serialize a vector to bytes."""
        v = self._validate_embedding(vector)
        return sqlite_vec.serialize_float32(v)

    def add_chunk(self, chunk_id: str, entity_id: str, entity_name: str,
                  section: str, content: str, embedding: np.ndarray,
                  position: int = 0) -> None:
        """Add single chunk with embedding (already computed externally)"""
        serialized = self._serialize_vec(embedding)

        # Delete previous vec_items row for this chunk_id (if any) to prevent orphans
        self.conn.execute("""
            DELETE FROM vec_items 
            WHERE rowid IN (
                SELECT v.rowid FROM vec_items v
                JOIN vec_map m ON v.rowid = m.rowid
                WHERE m.chunk_id = ?
            )
        """, (chunk_id,))

        # Upsert chunks + vec_map
        self.conn.execute(
            """INSERT OR REPLACE INTO chunks 
               (id, entity_id, entity_name, section, content, embedding, position)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, entity_id, entity_name, section, content,
             serialized, position),
        )

        cursor = self.conn.execute("SELECT last_insert_rowid()")
        rowid = cursor.fetchone()[0]

        self.conn.execute(
            """INSERT OR REPLACE INTO vec_map (rowid, chunk_id)
               VALUES (?, ?)""",
            (rowid, chunk_id),
        )

        # INSERT OR REPLACE into vec_items (idempotent)
        self.conn.execute(
            """INSERT OR REPLACE INTO vec_items (rowid, embedding)
               VALUES (?, ?)""",
            (rowid, serialized),
        )

        self.conn.commit()

    def add_chunks_batch(self, chunks: List[dict]) -> None:
        """Batch-add chunks with a single rowid lookup query."""
        if not chunks:
            return

        # Serialize once per chunk — avoids double validation
        chunk_data = []
        for c in chunks:
            emb = self._validate_embedding(c["embedding"])
            ser = sqlite_vec.serialize_float32(emb)
            chunk_data.append({
                "chunk_id": c["chunk_id"],
                "entity_id": c["entity_id"],
                "entity_name": c["entity_name"],
                "section": c["section"],
                "content": c["content"],
                "position": c.get("position", 0),
                "serialized": ser,
            })

        # Upsert all chunks
        self.conn.executemany(
            """INSERT OR REPLACE INTO chunks 
               (id, entity_id, entity_name, section, content, embedding, position)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(c["chunk_id"], c["entity_id"], c["entity_name"],
              c["section"], c["content"], c["serialized"], c["position"])
             for c in chunk_data],
        )
        self.conn.commit()

        # Single batch query for all rowids (fixes N+1)
        chunk_ids = [c["chunk_id"] for c in chunk_data]
        placeholders = ",".join("?" * len(chunk_ids))
        id_to_rowid = {
            r["id"]: r["rowid"]
            for r in self.conn.execute(
                f"SELECT id, rowid FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        }

        # Build map/vec rows from the single lookup
        map_rows = []
        vec_rows = []
        for c in chunk_data:
            rowid = id_to_rowid.get(c["chunk_id"])
            if rowid is not None:
                map_rows.append((rowid, c["chunk_id"]))
                vec_rows.append((rowid, c["serialized"]))

        if map_rows:
            self.conn.executemany(
                """INSERT OR REPLACE INTO vec_map (rowid, chunk_id)
                   VALUES (?, ?)""",
                map_rows,
            )
            self.conn.executemany(
                """INSERT OR REPLACE INTO vec_items (rowid, embedding)
                   VALUES (?, ?)""",
                vec_rows,
            )

        self.conn.commit()
        print(f"   [OK] Indexed {len(chunks)} chunks (SQLite-Vec)")

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               min_score: float = 0.0) -> List[SearchResult]:
        """Semantic search using sqlite-vec extension"""
        if top_k <= 0:
            return []

        query_vec = self._validate_embedding(query_embedding)
        serialized_q = sqlite_vec.serialize_float32(query_vec)

        # Use score alias in ORDER BY — avoids redundant vec_distance_cosine call
        cursor = self.conn.execute("""
            SELECT 
                c.id,
                c.entity_id,
                c.entity_name,
                c.section,
                c.content,
                (1.0 - vec_distance_cosine(v.embedding, ?) / 2.0) as score
            FROM vec_items v
            JOIN vec_map m ON v.rowid = m.rowid
            JOIN chunks c ON m.chunk_id = c.id
            ORDER BY score DESC
            LIMIT ?
        """, (serialized_q, top_k))

        results = []
        for row in cursor.fetchall():
            score = float(row["score"])
            if score >= min_score:
                results.append(SearchResult(
                    chunk_id=row["id"],
                    entity_id=row["entity_id"],
                    entity_name=row["entity_name"],
                    section=row["section"],
                    content=row["content"],
                    score=score,
                ))

        return results

    def search_by_entity(self, entity_id: str) -> List[dict]:
        """Get all chunks for specific entity"""
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE entity_id = ? ORDER BY position",
            (entity_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Store statistics"""
        total = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        entities = self.conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM chunks"
        ).fetchone()[0]
        return {
            "total_chunks": total,
            "total_entities": entities,
            "backend_type": "sqlite_vec",
            "db_path": self.db_path,
        }

    def get_indexed_entity_names(self) -> set:
        """Return all entity names currently stored in this backend."""
        rows = self.conn.execute(
            "SELECT DISTINCT entity_name FROM chunks"
        ).fetchall()
        return {r["entity_name"] for r in rows}

    def delete_entity(self, entity_id: str) -> None:
        """Remove all chunks for a given entity."""
        self.conn.execute("""
            DELETE FROM vec_items 
            WHERE rowid IN (
                SELECT v.rowid 
                FROM vec_items v
                JOIN vec_map m ON v.rowid = m.rowid
                JOIN chunks c ON m.chunk_id = c.id
                WHERE c.entity_id = ?
            )
        """, (entity_id,))

        self.conn.execute("""
            DELETE FROM vec_map
            WHERE chunk_id IN (
                SELECT id FROM chunks WHERE entity_id = ?
            )
        """, (entity_id,))

        self.conn.execute(
            "DELETE FROM chunks WHERE entity_id = ?", (entity_id,)
        )
        self.conn.commit()

    def close(self) -> None:
        """Close database connection"""
        self.conn.close()
