"""
YouTube Transcript Ingestion & Speech Chunker
=============================================
Fetches captions and video metadata from YouTube without requiring API keys,
chunks transcripts into speech-boundary-aware passages with start/end timestamps,
and formats canonical deep links (e.g. https://youtube.com/watch?v=...&t=318s).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    VideoUnplayable,
    AgeRestricted,
    IpBlocked,
    RequestBlocked,
    TranslationLanguageNotAvailable,
    YouTubeRequestFailed,
    YouTubeTranscriptApiException,
)

from src.utils.errors import DocumentProcessingError

logger = logging.getLogger(__name__)

# Regex patterns matching standard YouTube URLs, short URLs, embeds, and shorts
_YT_REGEX = re.compile(
    r"(?:https?:\/\/)?(?:www\.|m\.)?(?:youtube\.com\/(?:watch\?.*v=|embed\/|shorts\/|v\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
)
_RAW_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """
    Extract the canonical 11-character YouTube video ID from a URL or raw ID.

    Supports:
        - https://www.youtube.com/watch?v=dQw4w9WgXcQ
        - https://youtu.be/dQw4w9WgXcQ
        - https://www.youtube.com/embed/dQw4w9WgXcQ
        - https://www.youtube.com/shorts/dQw4w9WgXcQ
        - dQw4w9WgXcQ
    """
    cleaned = (url_or_id or "").strip()
    if _RAW_ID_REGEX.match(cleaned):
        return cleaned

    match = _YT_REGEX.search(cleaned)
    if match:
        return match.group(1)

    raise ValueError(f"Invalid YouTube URL or Video ID: '{url_or_id}'")


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into MM:SS or HH:MM:SS string.
    Examples:
        75.0   -> "01:15"
        318.0  -> "05:18"
        3725.0 -> "01:02:05"
    """
    total_seconds = int(max(0.0, seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fetch_video_metadata(video_id: str) -> Dict[str, Any]:
    """
    Fetch public metadata for a YouTube video using YouTube's oEmbed endpoint.
    Retrieves title and channel/author without an API key.

    Note: oEmbed does not return duration; duration is estimated from captions.
    """
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    oembed_url = f"https://www.youtube.com/oembed?url={canonical_url}&format=json"

    metadata = {
        "video_id": video_id,
        "video_url": canonical_url,
        "title": f"YouTube Video ({video_id})",
        "channel": "YouTube Creator",
    }

    try:
        req = urllib.request.Request(
            oembed_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                metadata["title"] = data.get("title") or metadata["title"]
                metadata["channel"] = data.get("author_name") or metadata["channel"]
                logger.info(
                    "Fetched oEmbed metadata for video '%s': '%s' by '%s'",
                    video_id, metadata["title"], metadata["channel"],
                )
    except Exception as exc:
        logger.warning(
            "Could not fetch oEmbed metadata for video '%s' (%s) — using fallback title",
            video_id, exc,
        )

    return metadata


def _get_transcript_list(video_id: str):
    """Retrieve TranscriptList handling both older and modern versions of youtube-transcript-api."""
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        return YouTubeTranscriptApi.list_transcripts(video_id)
    # Modern youtube-transcript-api (>=1.0)
    api = YouTubeTranscriptApi()
    return api.list(video_id)


def fetch_transcript(
    video_id: str,
    preferred_languages: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Retrieve transcript segments using youtube-transcript-api.
    Prefers manually created (creator) captions first, falling back to auto-generated captions.

    Returns:
        Tuple of:
          - raw_segments: List of {'text': str, 'start': float, 'duration': float}
          - transcript_origin: 'creator' | 'auto'
          - language_code: str (e.g. 'en')

    Raises:
        DocumentProcessingError with friendly, clear diagnostic message on failure.
    """
    languages = preferred_languages or ["en", "en-US", "en-GB", "en-CA"]
    logger.info("Fetching transcripts for video '%s' (preferred=%s)", video_id, languages)

    try:
        transcript_list = _get_transcript_list(video_id)

        chosen_transcript = None
        origin = "creator"

        # 1. Try manual (creator) captions in preferred languages
        for lang in languages:
            try:
                chosen_transcript = transcript_list.find_manually_created_transcript([lang])
                origin = "creator"
                break
            except Exception:
                continue

        # 2. Try any other manual (creator) caption
        if chosen_transcript is None:
            manual_transcripts = [t for t in transcript_list if not t.is_generated]
            if manual_transcripts:
                chosen_transcript = manual_transcripts[0]
                origin = "creator"

        # 3. Try auto-generated captions in preferred languages
        if chosen_transcript is None:
            for lang in languages:
                try:
                    chosen_transcript = transcript_list.find_generated_transcript([lang])
                    origin = "auto"
                    break
                except Exception:
                    continue

        # 4. Try any auto-generated caption
        if chosen_transcript is None:
            generated_transcripts = [t for t in transcript_list if t.is_generated]
            if generated_transcripts:
                chosen_transcript = generated_transcripts[0]
                origin = "auto"

        if chosen_transcript is None:
            raise DocumentProcessingError(
                f"No suitable captions found for video '{video_id}'."
            )

        logger.info(
            "Found %s caption for video '%s' in language '%s'",
            origin, video_id, chosen_transcript.language_code,
        )
        raw_fetched = chosen_transcript.fetch()
        if hasattr(raw_fetched, "to_raw_data") and callable(raw_fetched.to_raw_data):
            raw_segments = raw_fetched.to_raw_data()
        else:
            raw_segments = []
            for s in raw_fetched:
                if isinstance(s, dict):
                    raw_segments.append({
                        "text": str(s.get("text", "")),
                        "start": float(s.get("start", 0.0)),
                        "duration": float(s.get("duration", 0.0)),
                    })
                else:
                    raw_segments.append({
                        "text": str(getattr(s, "text", "")),
                        "start": float(getattr(s, "start", 0.0)),
                        "duration": float(getattr(s, "duration", 0.0)),
                    })
        return raw_segments, origin, chosen_transcript.language_code

    except TranscriptsDisabled:
        raise DocumentProcessingError(
            f"Captions are disabled for YouTube video '{video_id}'."
        )
    except NoTranscriptFound:
        raise DocumentProcessingError(
            f"No captions available for YouTube video '{video_id}' (neither creator nor auto-generated)."
        )
    except (VideoUnavailable, VideoUnplayable):
        raise DocumentProcessingError(
            f"YouTube video '{video_id}' is unavailable (may be private, removed, or region-restricted)."
        )
    except AgeRestricted:
        raise DocumentProcessingError(
            f"YouTube video '{video_id}' is age-restricted and its captions cannot be retrieved without login."
        )
    except (IpBlocked, RequestBlocked, YouTubeRequestFailed):
        raise DocumentProcessingError(
            "YouTube temporarily blocked or rate-limited transcript requests from this IP. Please try again later."
        )
    except YouTubeTranscriptApiException as exc:
        raise DocumentProcessingError(
            f"Failed to retrieve transcript for video '{video_id}': {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error fetching YouTube transcript")
        raise DocumentProcessingError(
            f"Could not retrieve captions for YouTube video '{video_id}': {exc}"
        ) from exc


def chunk_transcript(
    raw_segments: List[Dict[str, Any]],
    video_id: str,
    video_title: str,
    video_url: str,
    channel: Optional[str] = None,
    transcript_origin: str = "creator",
    language: str = "en",
    target_duration: float = 45.0,
    overlap_duration: float = 10.0,
    max_duration: float = 90.0,
) -> List[Dict[str, Any]]:
    """
    Speech-aware chunking for YouTube transcript slices.
    Groups sequential subtitle snippets by time duration (~45s window, ~10s overlap)
    and snaps to punctuation and speech pauses.

    Every chunk records its start_seconds, end_seconds, and canonical timestamp_url.
    """
    if not raw_segments:
        return []

    # Clean raw subtitle text
    cleaned_segments = []
    for seg in raw_segments:
        text_val = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        start_val = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        dur_val = seg.get("duration", 0.0) if isinstance(seg, dict) else getattr(seg, "duration", 0.0)

        text = re.sub(r"\s+", " ", str(text_val or "")).strip()
        if text:
            st = float(start_val or 0.0)
            du = float(dur_val or 0.0)
            cleaned_segments.append({
                "text": text,
                "start": st,
                "duration": du,
                "end": st + du,
            })

    if not cleaned_segments:
        return []

    chunks: List[Dict[str, Any]] = []
    total_segments = len(cleaned_segments)
    current_idx = 0
    chunk_index = 0
    char_offset = 0

    while current_idx < total_segments:
        window_start_time = cleaned_segments[current_idx]["start"]
        window_segments: List[Dict[str, Any]] = []
        next_start_idx = None

        idx = current_idx
        while idx < total_segments:
            seg = cleaned_segments[idx]
            window_segments.append(seg)

            current_duration = seg["end"] - window_start_time

            # Identify if next segment qualifies for overlap threshold
            if next_start_idx is None and current_duration >= (target_duration - overlap_duration):
                next_start_idx = idx

            # Boundary checks:
            # 1. Punctuation boundary if past target duration
            ends_with_punct = bool(re.search(r"[.!?]$", seg["text"]))
            # 2. Significant speech pause (> 1.2s silence before next segment)
            pause_after = False
            if idx + 1 < total_segments:
                pause_after = (cleaned_segments[idx + 1]["start"] - seg["end"]) > 1.2

            if current_duration >= target_duration and (ends_with_punct or pause_after):
                break

            # Hard upper duration limit
            if current_duration >= max_duration:
                break

            idx += 1

        if not window_segments:
            break

        chunk_start_sec = window_segments[0]["start"]
        chunk_end_sec = window_segments[-1]["end"]
        chunk_text = " ".join(s["text"] for s in window_segments).strip()

        start_int = int(chunk_start_sec)
        timestamp_url = f"https://www.youtube.com/watch?v={video_id}&t={start_int}s"
        source_label = f"{video_title} [{format_timestamp(chunk_start_sec)} - {format_timestamp(chunk_end_sec)}]"

        chunk_entry = {
            "doc_id": f"yt_{video_id}",
            "text": chunk_text,
            "headers": {"H1": video_title, "H2": None, "H3": None},
            "page_num": None,
            "chunk_index": chunk_index,
            "block_type": "transcript",
            "block_metadata": {
                "format": "youtube_transcript",
                "segment_count": len(window_segments),
            },
            "source_type": "youtube",
            "video_id": video_id,
            "video_url": video_url,
            "video_title": video_title,
            "channel": channel or "YouTube Creator",
            "start_seconds": round(chunk_start_sec, 2),
            "end_seconds": round(chunk_end_sec, 2),
            "timestamp_url": timestamp_url,
            "transcript_origin": transcript_origin,
            "language": language,
            "source_label": source_label,
            "original_char_count": len(chunk_text),
            "char_position_in_doc": char_offset,
        }
        chunks.append(chunk_entry)

        char_offset += len(chunk_text) + 1
        chunk_index += 1

        # Advance window
        if next_start_idx is not None and next_start_idx > current_idx and next_start_idx < total_segments:
            current_idx = next_start_idx
        else:
            current_idx = idx + 1

    logger.info(
        "Transcript chunking for video '%s': %d raw segments -> %d speech chunks",
        video_id, len(raw_segments), len(chunks),
    )
    return chunks
