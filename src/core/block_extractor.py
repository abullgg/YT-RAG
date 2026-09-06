"""
Block Extractor
---------------
Identifies and preserves "atomic" content blocks — tables and code snippets —
that must never be split mid-structure by the recursive character splitter.

Flow
~~~~
1. Call ``BlockExtractor.extract(text)`` **before** header normalisation.
   The method returns the text with each detected block replaced by a
   placeholder token ``[BLOCK_0]``, ``[BLOCK_1]``, … and a ``blocks`` list
   that stores the original content + metadata.

2. Run the normal 2-pass chunking on the placeholder-inclusive text.

3. After chunking, call ``BlockExtractor.restore(chunk, blocks)`` on every
   resulting chunk to swap placeholders back to real content.

Detected block types
~~~~~~~~~~~~~~~~~~~~
- **Markdown tables** — sequences of ``| … |`` rows (≥ 2 rows)
- **HTML tables**     — ``<table>…</table>`` spans
- **Triple-backtick code fences** — `` ``` … ``` ``
- **HTML code/pre blocks** — ``<pre>…</pre>`` / ``<code>…</code>``
- **Indented code** — ≥4 consecutive lines each starting with 4 spaces or 1 tab
  (only when flanked by blank lines to avoid false-positives in prose)

Oversized blocks (>2000 chars) are kept whole and emit a WARNING so the caller
knows a chunk has exceeded the normal size budget.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wrapped block format written into the text
# ---------------------------------------------------------------------------

_PLACEHOLDER_FMT = "[BLOCK_{idx}]"
_WRAP_OPEN  = '[ATOMIC_BLOCK type="{btype}" format="{fmt}"{lang_attr}]'
_WRAP_CLOSE = "[/ATOMIC_BLOCK]"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedBlock:
    """One atomic block pulled from the document."""
    index: int                          # matches placeholder index
    block_type: str                     # "table" | "code"
    format: str                         # "markdown" | "html" | "ascii" | "fenced" | "indented"
    language: Optional[str]             # code language hint (may be None)
    raw_content: str                    # the original text (no placeholder)
    wrapped_content: str                # content wrapped in ATOMIC_BLOCK markers
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Return value of BlockExtractor.extract()."""
    text_with_placeholders: str
    blocks: List[ExtractedBlock]


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# ── Markdown table ──────────────────────────────────────────────────────────
# Two or more consecutive lines that start and end with |
# (the separator row `|---|` is optional but common)
_RE_MD_TABLE = re.compile(
    r"((?:\|[^\n]*\|[ \t]*\n){2,})",
    re.MULTILINE,
)

# ── HTML table ──────────────────────────────────────────────────────────────
_RE_HTML_TABLE = re.compile(
    r"(<table[\s\S]*?</table>)",
    re.IGNORECASE | re.DOTALL,
)

# ── Triple-backtick code fence ───────────────────────────────────────────────
# Captures optional language specifier on the opening fence line
_RE_BACKTICK_FENCE = re.compile(
    r"(```(\w*)\n[\s\S]*?```)",
    re.MULTILINE,
)

# ── HTML <pre> / <code> ──────────────────────────────────────────────────────
_RE_HTML_PRE = re.compile(
    r"(<(?:pre|code)[\s\S]*?</(?:pre|code)>)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# BlockExtractor
# ---------------------------------------------------------------------------

class BlockExtractor:
    """
    Detects and extracts atomic blocks from document text.

    Args:
        max_split_size: Blocks larger than this emit a WARNING but are still
            kept whole. Should match the RecursiveCharacterTextSplitter chunk
            size (default 2000).
    """

    def __init__(self, max_split_size: int = 2000) -> None:
        self.max_split_size = max_split_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> ExtractionResult:
        """
        Locate every atomic block in *text*, replace each with a placeholder,
        and return the modified text together with block metadata.

        Detection order matters — patterns are applied from highest to lowest
        specificity to avoid double-extraction:
            1. Triple-backtick fences (most specific, unambiguous markers)
            2. HTML <pre>/<code>
            3. HTML <table>
            4. Markdown tables
            5. Indented code blocks (most ambiguous, last)

        Args:
            text: Raw document text.

        Returns:
            :class:`ExtractionResult`
        """
        blocks: List[ExtractedBlock] = []
        # Working copy — we progressively replace matched spans
        working = text

        # Track already-replaced regions so later passes don't touch them.
        # We achieve this simply by running each regex on the *working* copy
        # which already has placeholders for earlier matches.

        working = self._extract_pattern(
            working, _RE_BACKTICK_FENCE, "code", "fenced", blocks,
            lang_fn=lambda m: m.group(2) or None,
        )
        working = self._extract_pattern(
            working, _RE_HTML_PRE, "code", "html", blocks,
        )
        working = self._extract_pattern(
            working, _RE_HTML_TABLE, "table", "html", blocks,
        )
        working = self._extract_pattern(
            working, _RE_MD_TABLE, "table", "markdown", blocks,
        )
        working = self._extract_indented_code(working, blocks)

        if blocks:
            logger.info("BlockExtractor: extracted %d atomic block(s)", len(blocks))

        return ExtractionResult(text_with_placeholders=working, blocks=blocks)

    @staticmethod
    def restore(chunk_text: str, blocks: List[ExtractedBlock]) -> str:
        """
        Replace ``[BLOCK_N]`` placeholders inside *chunk_text* with the
        wrapped content of the corresponding :class:`ExtractedBlock`.

        Args:
            chunk_text: A single chunk (possibly containing placeholder tokens).
            blocks:     The block list returned by :meth:`extract`.

        Returns:
            The chunk text with all placeholders restored.
        """
        for block in blocks:
            placeholder = _PLACEHOLDER_FMT.format(idx=block.index)
            if placeholder in chunk_text:
                chunk_text = chunk_text.replace(placeholder, block.wrapped_content)
        return chunk_text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_pattern(
        self,
        text: str,
        pattern: re.Pattern,
        block_type: str,
        fmt: str,
        blocks: List[ExtractedBlock],
        lang_fn=None,
    ) -> str:
        """Generic regex extraction pass."""
        def _replacer(match: re.Match) -> str:
            raw = match.group(1)
            lang = lang_fn(match) if lang_fn else None
            block = self._make_block(len(blocks), block_type, fmt, lang, raw)
            blocks.append(block)
            return _PLACEHOLDER_FMT.format(idx=block.index)

        return pattern.sub(_replacer, text)

    def _extract_indented_code(self, text: str, blocks: List[ExtractedBlock]) -> str:
        """
        Find runs of ≥4 consecutive lines each starting with 4 spaces or a tab,
        flanked by blank lines (or document boundaries).
        """
        lines = text.split("\n")
        result_lines: List[str] = []
        i = 0
        while i < len(lines):
            # Must be preceded by blank (or start of doc)
            prev_blank = i == 0 or lines[i - 1].strip() == ""
            if prev_blank and self._is_indented_code_line(lines[i]):
                # Collect the run
                run_start = i
                while i < len(lines) and (
                    self._is_indented_code_line(lines[i]) or lines[i].strip() == ""
                ):
                    i += 1
                # Trim trailing blank lines from the run
                run_end = i
                while run_end > run_start and lines[run_end - 1].strip() == "":
                    run_end -= 1

                # Only qualify if ≥4 non-blank indented lines
                non_blank = [l for l in lines[run_start:run_end] if l.strip()]
                if len(non_blank) >= 4:
                    raw = "\n".join(lines[run_start:run_end])
                    block = self._make_block(len(blocks), "code", "indented", None, raw)
                    blocks.append(block)
                    result_lines.append(_PLACEHOLDER_FMT.format(idx=block.index))
                    continue
                else:
                    # Not enough lines — emit as-is
                    result_lines.extend(lines[run_start:i])
                    continue
            result_lines.append(lines[i])
            i += 1
        return "\n".join(result_lines)

    def _make_block(
        self,
        index: int,
        block_type: str,
        fmt: str,
        language: Optional[str],
        raw: str,
    ) -> ExtractedBlock:
        """Build an ExtractedBlock with wrapped content and emit size warning."""
        lang_attr = f' language="{language}"' if language else ""
        open_tag = _WRAP_OPEN.format(btype=block_type, fmt=fmt, lang_attr=lang_attr)
        wrapped = f"{open_tag}\n{raw}\n{_WRAP_CLOSE}"

        if len(raw) > self.max_split_size:
            logger.warning(
                "Atomic block #%d (%s/%s) is %d chars — exceeds split size %d. "
                "Block kept whole; context budget may be exceeded.",
                index, block_type, fmt, len(raw), self.max_split_size,
            )

        return ExtractedBlock(
            index=index,
            block_type=block_type,
            format=fmt,
            language=language,
            raw_content=raw,
            wrapped_content=wrapped,
            metadata={"char_count": len(raw)},
        )

    @staticmethod
    def _is_indented_code_line(line: str) -> bool:
        """True if the line starts with 4+ spaces or a tab."""
        return line.startswith("    ") or line.startswith("\t")
