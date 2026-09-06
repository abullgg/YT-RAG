"""
Persistent Storage
------------------
Saves and loads the RAG application state to disk so indexed documents
survive server restarts.

Files written to ./data/:
  faiss_index.bin — the serialised FAISS index
  metadata.json   — {doc_id: metadata} for all indexed documents
  chunks.pkl      — FAISS position → ChunkMetadata dict mapping

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~
Old chunks.pkl files contain ``{int: (doc_id, text)}`` tuples.
The loader detects legacy format and migrates entries on-the-fly via
``_migrate_legacy_entry()`` from faiss_index.py — no data loss.

Dimension guard
~~~~~~~~~~~~~~~
On load, the stored index dimension is checked against the configured
EMBEDDING_DIMENSION. If they differ (e.g. after switching embedding models),
the stale files are deleted and the server starts with an empty index.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import faiss

from src.core.config import settings
from src.vector_store.faiss_index import _migrate_legacy_entry

logger = logging.getLogger(__name__)


class PersistentStorage:
    """
    Saves and loads RAG state (FAISS index, document metadata, chunk store).

    Never raises on load failures — always returns empty state so the server
    starts cleanly even when files are missing or corrupted.

    Args:
        data_dir: Directory to store persistence files. Defaults to "data".
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.faiss_path = self.data_dir / "faiss_index.bin"
        self.meta_path  = self.data_dir / "metadata.json"
        self.chunks_path = self.data_dir / "chunks.pkl"

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create data directory: %s", exc)

    # ------------------------------------------------------------------ #
    #  Save
    # ------------------------------------------------------------------ #

    def save_state(
        self,
        faiss_index: Optional[faiss.IndexFlatIP],
        indexed_documents: Dict[str, dict],
        chunk_store: Dict[int, Any],
    ) -> bool:
        """
        Write FAISS index, document metadata, and chunk store to disk.

        The chunk store is always serialised in the *new* rich dict format.
        Any legacy tuple entries are migrated before saving so the file on
        disk is always up-to-date.

        Args:
            faiss_index:        The active FAISS index (skipped if None).
            indexed_documents:  {doc_id: metadata} registry.
            chunk_store:        FAISS position → ChunkMetadata mapping.

        Returns:
            True if all files were saved successfully.
        """
        try:
            if faiss_index is not None:
                faiss.write_index(faiss_index, str(self.faiss_path))

            with open(self.meta_path, "w", encoding="utf-8") as fh:
                json.dump(indexed_documents, fh, indent=2)

            # Migrate any legacy tuples before pickling
            migrated_store = {
                pos: _migrate_legacy_entry(entry)
                for pos, entry in chunk_store.items()
            }
            with open(self.chunks_path, "wb") as fh:
                pickle.dump(migrated_store, fh, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(
                "RAG state saved to '%s' (%d chunk entries)",
                self.data_dir,
                len(chunk_store),
            )
            return True

        except Exception as exc:
            logger.error("Failed to save state: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  Load
    # ------------------------------------------------------------------ #

    def load_state(
        self,
    ) -> Tuple[Optional[faiss.IndexFlatIP], Dict[str, dict], Dict[int, Any]]:
        """
        Load persisted RAG state from disk.

        If the stored FAISS index dimension doesn't match EMBEDDING_DIMENSION
        (which happens when you change the embedding model), all stale files are
        deleted and empty state is returned.

        Handles both new-format (dict) and legacy-format (tuple) chunk stores
        transparently.

        Returns:
            Tuple of (faiss_index, indexed_documents, chunk_store).
            Any missing or unreadable file returns its empty default.
        """
        faiss_index = None
        indexed_documents: Dict[str, dict] = {}
        chunk_store: Dict[int, Any] = {}

        try:
            if self.faiss_path.exists():
                faiss_index = faiss.read_index(str(self.faiss_path))
                logger.info(
                    "Loaded FAISS index from '%s' (dim=%d, vectors=%d)",
                    self.faiss_path, faiss_index.d, faiss_index.ntotal,
                )

                if faiss_index.d != settings.EMBEDDING_DIMENSION:
                    logger.warning(
                        "Stored index dimension (%d) doesn't match configured dimension (%d). "
                        "Clearing stale index — all documents must be re-uploaded.",
                        faiss_index.d, settings.EMBEDDING_DIMENSION,
                    )
                    self.clear_state()
                    return None, {}, {}

            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as fh:
                    indexed_documents = json.load(fh)
                logger.info(
                    "Loaded metadata from '%s' (%d documents)",
                    self.meta_path, len(indexed_documents),
                )

            if self.chunks_path.exists():
                with open(self.chunks_path, "rb") as fh:
                    raw_store = pickle.load(fh)

                # Migrate any legacy entries on load
                chunk_store = {
                    pos: _migrate_legacy_entry(entry)
                    for pos, entry in raw_store.items()
                }
                legacy_count = sum(
                    1 for e in raw_store.values() if isinstance(e, tuple)
                )
                if legacy_count:
                    logger.info(
                        "Migrated %d legacy chunk store entries to rich dict format",
                        legacy_count,
                    )
                logger.info(
                    "Loaded chunk store from '%s' (%d entries)",
                    self.chunks_path, len(chunk_store),
                )

        except Exception as exc:
            logger.error(
                "Failed to load state from disk — starting with empty state: %s", exc
            )
            return None, {}, {}

        return faiss_index, indexed_documents, chunk_store

    # ------------------------------------------------------------------ #
    #  Clear
    # ------------------------------------------------------------------ #

    def clear_state(self) -> bool:
        """
        Delete all persisted state files.

        Use this when switching embedding models to force a clean re-index.

        Returns:
            True if the operation succeeded.
        """
        try:
            for path in (self.faiss_path, self.meta_path, self.chunks_path):
                if path.exists():
                    path.unlink()
            logger.info("Cleared all persisted state from '%s'", self.data_dir)
            return True
        except Exception as exc:
            logger.error("Failed to clear persisted state: %s", exc)
            return False
