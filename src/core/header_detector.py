"""
Header Detector
---------------
Converts ambiguous document headers (numbered sections, ALL-CAPS lines,
underline-style, title-cased isolated lines) to Markdown `#` syntax so that
MarkdownHeaderTextSplitter can segment the document correctly.

Every pattern is assigned a confidence score. Lines that don't reach the
configured threshold are left unchanged and logged at DEBUG level. This
prevents the false-positive headers that plagued the old 5-rule regex cascade.

Confidence levels
~~~~~~~~~~~~~~~~~
    1.00 — already Markdown (`#` prefix)           — always convert
    0.95 — numbered section  (`1.2 Methods`)        — always convert
    0.85 — underlined header  (`Title\\n---`)        — requires 3+ dashes/equals
                                                       and a non-empty line above
    0.70 — ALL-CAPS isolated  (`RESULTS`)           — requires ≥200 body chars
                                                       below before next heading
    0.60 — title-cased short  (`Data Collection`)  — preceded by blank line AND
                                                       followed by non-blank line

Default threshold: 0.75  (accepts Markdown, numbered, underlined; rejects the
two weakest patterns unless explicitly lowered).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectedHeader:
    """Record of one header that was (or could be) normalised."""
    line_num: int
    original: str
    normalized: str
    confidence: float
    accepted: bool          # True → line was converted; False → below threshold


@dataclass
class NormalizationResult:
    """Output of HeaderDetector.normalize()."""
    normalized_text: str
    detected_headers: List[DetectedHeader] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------

# 1. Numbered sections: "1.", "1.2", "2.1.3", "1)", all followed by a capital
_RE_NUMBERED = re.compile(r"^\d+(\.\d+)*[.)]\s+[A-Z]")

# 2. ALL-CAPS: 4+ chars, no trailing period, only letters/spaces/digits
_RE_ALL_CAPS = re.compile(r"^[A-Z][A-Z\s\d]{3,}$")

# 3. Underline rows: 3+ consecutive = or - (with no other characters)
_RE_UNDERLINE = re.compile(r"^[-=]{3,}$")

# 4. Title-cased: starts with uppercase, length < 60, no trailing punctuation
_RE_TITLE_CASE_BAD_END = re.compile(r"[,.:;!?]$")
_ARTICLE_STARTERS = ("The ", "A ", "An ", "the ", "a ", "an ")


def _is_title_case_candidate(stripped: str) -> bool:
    if not stripped or len(stripped) >= 60:
        return False
    if not stripped[0].isupper():
        return False
    if _RE_TITLE_CASE_BAD_END.search(stripped):
        return False
    if stripped.startswith(_ARTICLE_STARTERS):
        return False
    # Reject if the line contains mostly lowercase (= sentence, not a heading)
    words = stripped.split()
    if len(words) > 8:
        return False
    return True


# ---------------------------------------------------------------------------
# HeaderDetector
# ---------------------------------------------------------------------------

class HeaderDetector:
    """
    Confidence-gated header normalizer.

    Args:
        confidence_threshold: Lines whose pattern confidence is below this
            value are left unchanged. Default 0.75.
    """

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> NormalizationResult:
        """
        Scan *text* line by line, convert detected headers to `# …` Markdown,
        and return a :class:`NormalizationResult` with the modified text and
        an audit log of every decision.

        Args:
            text: Raw document text (plain or lightly structured).

        Returns:
            :class:`NormalizationResult`
        """
        lines = text.split("\n")
        output_lines: List[str] = []
        detected: List[DetectedHeader] = []
        skip_next = False

        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue  # consumed by underline handler on previous iteration

            stripped = line.strip()

            if not stripped:
                output_lines.append(line)
                continue

            result = self._classify_line(lines, i, stripped)

            if result is None:
                # No pattern matched at all
                output_lines.append(line)
                continue

            confidence, normalized_line, is_underline = result
            accepted = confidence >= self.confidence_threshold

            detected.append(
                DetectedHeader(
                    line_num=i + 1,          # 1-indexed for human readability
                    original=stripped,
                    normalized=normalized_line.strip(),
                    confidence=confidence,
                    accepted=accepted,
                )
            )

            if accepted:
                output_lines.append(normalized_line)
                if is_underline:
                    skip_next = True   # drop the `---` / `===` line
            else:
                logger.debug(
                    "Header skipped (conf=%.2f < threshold=%.2f): %r",
                    confidence,
                    self.confidence_threshold,
                    stripped,
                )
                output_lines.append(line)

        return NormalizationResult(
            normalized_text="\n".join(output_lines),
            detected_headers=detected,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_line(
        self,
        lines: List[str],
        i: int,
        stripped: str,
    ) -> Optional[Tuple[float, str, bool]]:
        """
        Try every pattern in descending confidence order.

        Returns:
            ``(confidence, replacement_line, is_underline)`` or ``None`` if no
            pattern matches.
        """
        # ── Pattern 0: already Markdown ─────────────────────────────────────
        if stripped.startswith("#"):
            return 1.0, stripped, False

        # ── Pattern 1: numbered section ─────────────────────────────────────
        if _RE_NUMBERED.match(stripped):
            return 0.95, f"# {stripped}", False

        # ── Pattern 2: underlined header ────────────────────────────────────
        # Check if the *next* line is a pure underline row
        if i < len(lines) - 1:
            next_stripped = lines[i + 1].strip()
            if _RE_UNDERLINE.match(next_stripped):
                # Require the line above *this* line to be non-empty
                # (visual separators like a row of dashes after a blank line
                # should not be treated as headers).
                prev_non_empty = i > 0 and lines[i - 1].strip() != ""
                if prev_non_empty or i == 0:
                    return 0.85, f"# {stripped}", True

        # ── Pattern 3: ALL-CAPS isolated ────────────────────────────────────
        if _RE_ALL_CAPS.match(stripped) and not stripped.endswith("."):
            # Only accept if ≥200 chars of body text follow before next heading
            body_chars = self._chars_before_next_heading(lines, i + 1)
            if body_chars >= 200:
                return 0.70, f"# {stripped}", False

        # ── Pattern 4: title-cased isolated line ────────────────────────────
        if _is_title_case_candidate(stripped):
            preceded_by_blank = i == 0 or lines[i - 1].strip() == ""
            followed_by_body = (
                i < len(lines) - 1 and lines[i + 1].strip() != ""
            )
            if preceded_by_blank and followed_by_body:
                return 0.60, f"# {stripped}", False

        return None

    @staticmethod
    def _chars_before_next_heading(lines: List[str], start: int) -> int:
        """Count body characters between *start* and the next heading-like line."""
        count = 0
        for line in lines[start:]:
            s = line.strip()
            if s.startswith("#") or _RE_NUMBERED.match(s) or _RE_UNDERLINE.match(s):
                break
            count += len(s)
        return count
