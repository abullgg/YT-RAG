"""
Document Processor
------------------
Handles file ingestion (PDF / TXT): text and table extraction,
semantic chunking, and unique document ID generation.

Chunking pipeline (4 passes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
0. BlockExtractor  — detect and replace atomic tables / code blocks with
                     ``[BLOCK_N]`` placeholders before any splitting.
1. HeaderDetector  — confidence-gated normalization of ambiguous headers to
                     Markdown ``#`` syntax (replaces the old regex cascade).
2. MarkdownHeaderTextSplitter — semantic split on ``#``/``##``/``###``
                     boundaries; header breadcrumbs re-attached to each chunk.
3. RecursiveCharacterTextSplitter — size-enforcement safety net (configurable
                     per doc_type via CHUNK_SIZE_PRESETS).
4. Block restore   — swap ``[BLOCK_N]`` tokens back to wrapped ATOMIC_BLOCK
                     content in every final chunk.
"""

from __future__ import annotations

import io
import logging
import uuid
import re
from typing import Any, Dict, List, Optional

from fastapi import UploadFile
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from src.core.config import settings, CHUNK_SIZE_PRESETS, DocType
from src.core.header_detector import HeaderDetector, NormalizationResult
from src.core.block_extractor import BlockExtractor, ExtractionResult, ExtractedBlock
from src.utils.errors import DocumentProcessingError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChunkResult — richer return type from chunk_text()
# ---------------------------------------------------------------------------

class ChunkResult:
    """
    Carries the text content of a single chunk **plus** the metadata harvested
    during 2-pass splitting (header breadcrumbs, block type, sizes, etc.).
    These metadata fields are stored verbatim into the chunk store and returned
    with retrieval results.
    """

    __slots__ = (
        "text",
        "headers",
        "block_type",
        "block_metadata",
        "original_char_count",
        "char_position_in_doc",
        "chunk_index",
    )

    def __init__(
        self,
        text: str,
        headers: Optional[Dict[str, Optional[str]]] = None,
        block_type: str = "text",
        block_metadata: Optional[Dict[str, Any]] = None,
        original_char_count: int = 0,
        char_position_in_doc: int = 0,
        chunk_index: int = 0,
    ) -> None:
        self.text = text
        self.headers: Dict[str, Optional[str]] = headers or {"H1": None, "H2": None, "H3": None}
        self.block_type = block_type
        self.block_metadata = block_metadata
        self.original_char_count = original_char_count
        self.char_position_in_doc = char_position_in_doc
        self.chunk_index = chunk_index

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ChunkResult index={self.chunk_index} "
            f"len={len(self.text)} "
            f"block_type={self.block_type!r} "
            f"headers={self.headers}>"
        )


# ---------------------------------------------------------------------------
# DocumentProcessor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """Extracts text from uploaded files and splits it into semantic chunks."""

    def __init__(self) -> None:
        # Pass 1B: Semantic header splitting
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ],
            return_each_line=False,
        )
        # Header detector (confidence-gated; replaces old 5-rule regex cascade)
        self._header_detector = HeaderDetector(
            confidence_threshold=settings.HEADER_CONFIDENCE_THRESHOLD
        )
        # Block extractor (tables + code fences)
        self._block_extractor = BlockExtractor(max_split_size=settings.CHUNK_SIZE)

        logger.info("DocumentProcessor initialized with 4-pass chunking pipeline")

    # ------------------------------------------------------------------ #
    #  Text Extraction
    # ------------------------------------------------------------------ #

    async def extract_text(self, raw_bytes: bytes, filename: str, content_type: str) -> str:
        """
        Read the bytes and return its full text content.

        Supports:
            - application/pdf  → extracted via pdfplumber (layout-aware)
            - text/plain (.txt) → decoded as UTF-8 (latin-1 fallback)

        Args:
            raw_bytes: Raw file bytes loaded into memory.
            filename: File name string.
            content_type: MIME string.

        Returns:
            The extracted text as a single string.

        Raises:
            DocumentProcessingError: If the file type is unsupported or empty.
        """
        logger.info("Extracting text from '%s' (content_type=%s)", filename, content_type)

        if not raw_bytes:
            raise DocumentProcessingError(f"Uploaded file '{filename}' is empty.")

        if filename.lower().endswith(".pdf") or "pdf" in content_type:
            return self._extract_pdf(raw_bytes, filename)

        if filename.lower().endswith(".txt") or "text" in content_type:
            return self._extract_txt(raw_bytes, filename)

        raise DocumentProcessingError(
            f"Unsupported file type: '{filename}' ({content_type}). "
            "Only PDF and TXT files are accepted."
        )

    # ------------------------------------------------------------------ #
    #  Private Helpers — Extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _table_to_markdown(table: list) -> str:
        """
        Convert a pdfplumber table (list of rows, each a list of cell strings)
        into a GitHub-style markdown table string.

        Returns an empty string if the table is empty or contains only blank cells.
        """
        if not table:
            return ""

        cleaned: List[List[str]] = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table
        ]

        if all(all(cell == "" for cell in row) for row in cleaned):
            return ""

        col_count = max(len(row) for row in cleaned)
        padded = [row + [""] * (col_count - len(row)) for row in cleaned]

        lines: List[str] = []
        lines.append("| " + " | ".join(padded[0]) + " |")
        lines.append("| " + " | ".join(["---"] * col_count) + " |")
        for row in padded[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @staticmethod
    def _extract_pdf(raw_bytes: bytes, filename: str) -> str:
        """
        Parse PDF bytes and return the full text content.

        Strategy (per page):
        1. ``page.extract_text()``  — always runs first; preserves prose flow.
        2. ``page.extract_tables()`` — additionally extracts tables as Markdown
           and appends them after the page text. Failures are caught and ignored.
        """
        try:
            pages_text: List[str] = []
            num_pages = 0

            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                num_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    parts: List[str] = []

                    # 1. Prose text
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        parts.append(page_text)
                        logger.debug(
                            "Page %d/%d '%s': %d chars (text)",
                            page_num + 1, num_pages, filename, len(page_text),
                        )

                    # 2. Additive table extraction
                    try:
                        tables = page.extract_tables() or []
                        table_mds: List[str] = []
                        for table in tables:
                            md = DocumentProcessor._table_to_markdown(table)
                            if md:
                                table_mds.append(md)

                        if table_mds:
                            block = "[TABLE]\n" + "\n\n[TABLE]\n".join(table_mds)
                            parts.append(block)
                            logger.debug(
                                "Page %d/%d '%s': %d table(s) extracted as markdown",
                                page_num + 1, num_pages, filename, len(table_mds),
                            )
                    except Exception as tbl_exc:
                        logger.warning(
                            "Page %d/%d '%s': table extraction skipped (%s)",
                            page_num + 1, num_pages, filename, tbl_exc,
                        )

                    if parts:
                        pages_text.append("\n\n".join(parts))

            full_text = "\n\n".join(pages_text)
            logger.info("PDF '%s': %d pages, %d chars total", filename, num_pages, len(full_text))
            return full_text

        except Exception as exc:
            logger.error("Failed to parse PDF '%s': %s", filename, exc)
            raise DocumentProcessingError(f"Could not parse PDF '{filename}': {exc}") from exc

    @staticmethod
    def _extract_txt(raw_bytes: bytes, filename: str) -> str:
        """Decode plain-text bytes with a UTF-8 fallback to latin-1."""
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for '%s'; falling back to latin-1", filename)
            text = raw_bytes.decode("latin-1")

        logger.info("TXT '%s': %d chars extracted", filename, len(text))
        return text

    # ------------------------------------------------------------------ #
    #  Chunking Pipeline
    # ------------------------------------------------------------------ #

    def chunk_text(
        self,
        text: str,
        doc_type: DocType = "default",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        header_detection_enabled: Optional[bool] = None,
        header_confidence_threshold: Optional[float] = None,
    ) -> List[ChunkResult]:
        """
        4-pass semantic chunking pipeline.

        Pass 0 — BlockExtractor:
            Atomic tables and code blocks are replaced by ``[BLOCK_N]``
            placeholders so they are never split internally.

        Pass 1A — HeaderDetector:
            Confidence-gated header normalization converts ambiguous lines to
            Markdown ``#`` syntax. Can be disabled per call for narrative PDFs.

        Pass 1B — MarkdownHeaderTextSplitter:
            Splits on ``#``/``##``/``###`` boundaries. Header breadcrumbs
            (H1, H2, H3) are extracted and re-attached to each chunk.

        Pass 2 — RecursiveCharacterTextSplitter:
            Ensures no chunk exceeds ``chunk_size`` characters. Separator
            priority: ``\\n\\n`` → ``\\n`` → ``. `` → `` `` → char.

        Pass 3 — Block restore:
            ``[BLOCK_N]`` placeholders are swapped back to wrapped
            ``[ATOMIC_BLOCK]`` content inside every final chunk.

        Args:
            text: Full document text (from extract_text).
            doc_type: Preset key selecting a chunk size tuned to the content
                type. Ignored when ``chunk_size`` is provided explicitly.
            chunk_size: Override chunk size in characters.
            chunk_overlap: Override overlap in characters. Defaults to 20% of
                chunk_size when not provided.
            header_detection_enabled: Override the global
                ``HEADER_DETECTION_ENABLED`` setting for this call.
            header_confidence_threshold: Override the global
                ``HEADER_CONFIDENCE_THRESHOLD`` for this call.

        Returns:
            List of :class:`ChunkResult` objects (one per final chunk).

        Raises:
            DocumentProcessingError: On empty text or irrecoverable errors.
        """
        if not text or not text.strip():
            raise DocumentProcessingError("Cannot chunk empty text")

        # ── Resolve chunk parameters ──────────────────────────────────────
        effective_chunk_size = (
            chunk_size
            if chunk_size is not None
            else CHUNK_SIZE_PRESETS.get(doc_type, CHUNK_SIZE_PRESETS["default"])
        )
        effective_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else max(200, int(effective_chunk_size * 0.20))
        )
        use_header_detection = (
            header_detection_enabled
            if header_detection_enabled is not None
            else settings.HEADER_DETECTION_ENABLED
        )
        effective_threshold = (
            header_confidence_threshold
            if header_confidence_threshold is not None
            else settings.HEADER_CONFIDENCE_THRESHOLD
        )

        logger.info(
            "chunk_text() — doc_type=%s, chunk_size=%d, overlap=%d, "
            "header_detection=%s, threshold=%.2f",
            doc_type, effective_chunk_size, effective_overlap,
            use_header_detection, effective_threshold,
        )

        # Build a fresh splitter with the resolved sizes
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        try:
            # ── Pass 0: Block extraction ──────────────────────────────────
            extraction: ExtractionResult = self._block_extractor.extract(text)
            working_text = extraction.text_with_placeholders
            blocks = extraction.blocks
            if blocks:
                logger.info("Pass 0: extracted %d atomic block(s)", len(blocks))

            # ── Pass 1A: Header normalization ─────────────────────────────
            if use_header_detection:
                # Use the per-call threshold if it differs from the instance default
                if effective_threshold != self._header_detector.confidence_threshold:
                    detector = HeaderDetector(confidence_threshold=effective_threshold)
                else:
                    detector = self._header_detector
                norm_result: NormalizationResult = detector.normalize(working_text)
                working_text = norm_result.normalized_text
                accepted = sum(1 for h in norm_result.detected_headers if h.accepted)
                skipped = len(norm_result.detected_headers) - accepted
                logger.info(
                    "Pass 1A: %d headers detected (%d accepted, %d skipped below threshold)",
                    len(norm_result.detected_headers), accepted, skipped,
                )
            else:
                logger.info("Pass 1A: header detection disabled for this call")
                norm_result = None

            # ── Pass 1B: Semantic splitting ───────────────────────────────
            try:
                semantic_docs = self.markdown_splitter.split_text(working_text)
                logger.info("Pass 1B: MarkdownHeaderTextSplitter → %d semantic chunk(s)", len(semantic_docs))
            except Exception as e:
                logger.warning("Pass 1B: markdown splitting failed (%s) — using full text", e)
                semantic_docs = [{"page_content": working_text, "metadata": {}}]

            # ── Pass 2: Size enforcement + metadata harvest ───────────────
            char_pos = 0
            final_results: List[ChunkResult] = []

            for sem_chunk in semantic_docs:
                chunk_text_raw = getattr(sem_chunk, "page_content", None)
                if chunk_text_raw is None:
                    chunk_text_raw = (
                        sem_chunk.get("page_content", sem_chunk)
                        if isinstance(sem_chunk, dict)
                        else sem_chunk
                    )
                chunk_meta_raw = getattr(sem_chunk, "metadata", None)
                if chunk_meta_raw is None:
                    chunk_meta_raw = (
                        sem_chunk.get("metadata", {})
                        if isinstance(sem_chunk, dict)
                        else {}
                    )

                # Extract header breadcrumbs from LangChain metadata
                headers: Dict[str, Optional[str]] = {
                    "H1": chunk_meta_raw.get("H1"),
                    "H2": chunk_meta_raw.get("H2"),
                    "H3": chunk_meta_raw.get("H3"),
                }

                # Re-attach breadcrumb as readable prefix in the chunk text
                breadcrumb = " > ".join(v for v in headers.values() if v)
                if breadcrumb:
                    chunk_text_raw = f"[{breadcrumb}]\n{chunk_text_raw}"

                if len(chunk_text_raw) > effective_chunk_size:
                    logger.info(
                        "Pass 2: chunk (%d chars) exceeds %d — recursively splitting",
                        len(chunk_text_raw), effective_chunk_size,
                    )
                    sub_texts = recursive_splitter.split_text(chunk_text_raw)
                    for sub in sub_texts:
                        result = ChunkResult(
                            text=sub,
                            headers=headers,
                            block_type="text",
                            original_char_count=len(sub),
                            char_position_in_doc=char_pos,
                            chunk_index=len(final_results),
                        )
                        final_results.append(result)
                        char_pos += len(sub)
                else:
                    result = ChunkResult(
                        text=chunk_text_raw,
                        headers=headers,
                        block_type="text",
                        original_char_count=len(chunk_text_raw),
                        char_position_in_doc=char_pos,
                        chunk_index=len(final_results),
                    )
                    final_results.append(result)
                    char_pos += len(chunk_text_raw)

            # ── Pass 3: Block restore ─────────────────────────────────────
            if blocks:
                for cr in final_results:
                    restored = BlockExtractor.restore(cr.text, blocks)
                    if restored != cr.text:
                        # Identify which block(s) were injected
                        restored_blocks = [
                            b for b in blocks
                            if f"[BLOCK_{b.index}]" in cr.text
                        ]
                        cr.text = restored
                        cr.block_type = "atomic_block"
                        cr.block_metadata = {
                            "blocks": [
                                {"type": b.block_type, "format": b.format, "char_count": b.metadata.get("char_count")}
                                for b in restored_blocks
                            ]
                        }
                        cr.original_char_count = len(restored)

            # ── Diagnostic logging ────────────────────────────────────────
            logger.info("chunk_text() complete — %d final chunk(s)", len(final_results))
            if final_results:
                sizes = [len(r.text) for r in final_results]
                logger.info(
                    "Chunk sizes — min=%d, max=%d, avg=%d",
                    min(sizes), max(sizes), sum(sizes) // len(sizes),
                )

            return final_results

        except DocumentProcessingError:
            raise
        except Exception as exc:
            logger.error("Error chunking text: %s", exc)
            raise DocumentProcessingError(f"Failed to chunk text: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  ID Generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_document_id() -> str:
        """
        Generate a unique identifier for a newly ingested document.

        Returns:
            A UUID-4 string (e.g. ``'a3f8b1c2-...'``).
        """
        doc_id = str(uuid.uuid4())
        logger.debug("Generated document ID: %s", doc_id)
        return doc_id
