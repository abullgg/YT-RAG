"""
Deterministic unit tests for YouTube ingestion, speech chunking, and error handling.
Tested entirely with mock fixtures — zero live network requests.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.youtube import (
    extract_video_id,
    format_timestamp,
    chunk_transcript,
    fetch_transcript,
)
from src.utils.errors import DocumentProcessingError
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    IpBlocked,
)


class YouTubeIngestionTests(unittest.TestCase):

    # ── 1. URL & Video ID Parsing ───────────────────────────────────────────
    def test_extract_video_id_formats(self):
        expected_id = "dQw4w9WgXcQ"
        test_cases = [
            f"https://www.youtube.com/watch?v={expected_id}",
            f"https://youtube.com/watch?v={expected_id}&feature=share",
            f"https://youtu.be/{expected_id}",
            f"https://www.youtube.com/embed/{expected_id}",
            f"https://www.youtube.com/shorts/{expected_id}",
            f"http://youtube.com/watch?v={expected_id}",
            expected_id,  # Raw 11-char ID
        ]
        for url in test_cases:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), expected_id)

    def test_extract_video_id_invalid(self):
        invalid_urls = [
            "https://vimeo.com/123456",
            "not-a-youtube-url",
            "https://youtube.com/",
            "https://youtube.com/watch?v=too_short",
            "",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    extract_video_id(url)

    # ── 2. Timestamp Formatting ─────────────────────────────────────────────
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0.0), "00:00")
        self.assertEqual(format_timestamp(45.0), "00:45")
        self.assertEqual(format_timestamp(318.0), "05:18")
        self.assertEqual(format_timestamp(3665.0), "01:01:05")

    # ── 3. Speech-Aware Transcript Chunking & Timestamps ────────────────────
    def test_chunk_transcript_boundaries_and_overlap(self):
        # 10 segments of 6 seconds each = 60s total
        segments = [
            {"text": f"Sentence number {i}.", "start": float(i * 6), "duration": 6.0}
            for i in range(10)
        ]
        chunks = chunk_transcript(
            raw_segments=segments,
            video_id="test_video_1",
            video_title="Quantum Computing Basics",
            video_url="https://www.youtube.com/watch?v=test_video_1",
            channel="Tech Academy",
            transcript_origin="creator",
            language="en",
            target_duration=30.0,
            overlap_duration=10.0,
        )

        self.assertGreater(len(chunks), 1)
        first_chunk = chunks[0]
        self.assertEqual(first_chunk["video_id"], "test_video_1")
        self.assertEqual(first_chunk["start_seconds"], 0.0)
        self.assertIn("https://www.youtube.com/watch?v=test_video_1&t=0s", first_chunk["timestamp_url"])
        self.assertIn("Quantum Computing Basics [00:00 -", first_chunk["source_label"])

        # Second chunk should overlap with the first chunk
        second_chunk = chunks[1]
        self.assertLess(second_chunk["start_seconds"], first_chunk["end_seconds"])

    def test_repeated_text_chunks_preserve_distinct_timestamps(self):
        # Speaker repeats the exact same greeting at 10s and at 120s
        segments = [
            {"text": "Welcome to our tutorial.", "start": 10.0, "duration": 5.0},
            {"text": "In this section we cover vectors.", "start": 15.0, "duration": 40.0},
            {"text": "Welcome to our tutorial.", "start": 120.0, "duration": 5.0},
            {"text": "In this next section we cover matrices.", "start": 125.0, "duration": 40.0},
        ]
        chunks = chunk_transcript(
            raw_segments=segments,
            video_id="repeat_vid",
            video_title="Linear Algebra",
            video_url="https://www.youtube.com/watch?v=repeat_vid",
            target_duration=30.0,
            overlap_duration=5.0,
        )

        # Chunks containing "Welcome to our tutorial." must exist at 10s and 120s
        greeting_chunks = [c for c in chunks if "Welcome to our tutorial." in c["text"]]
        self.assertGreaterEqual(len(greeting_chunks), 2)
        starts = [c["start_seconds"] for c in greeting_chunks]
        self.assertIn(10.0, starts)
        self.assertIn(120.0, starts)
        urls = [c["timestamp_url"] for c in greeting_chunks]
        self.assertIn("https://www.youtube.com/watch?v=repeat_vid&t=10s", urls)
        self.assertIn("https://www.youtube.com/watch?v=repeat_vid&t=120s", urls)

    def test_chunk_transcript_with_dataclass_snippets(self):
        class DummySnippet:
            def __init__(self, text, start, duration):
                self.text = text
                self.start = start
                self.duration = duration

        snippets = [
            DummySnippet(f"Sentence {i}.", float(i * 10), 10.0)
            for i in range(5)
        ]
        chunks = chunk_transcript(
            raw_segments=snippets,
            video_id="dummy_vid",
            video_title="Dummy Title",
            video_url="https://www.youtube.com/watch?v=dummy_vid",
        )
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["video_id"], "dummy_vid")
        self.assertEqual(chunks[0]["start_seconds"], 0.0)

    # ── 4. Creator vs Auto Caption Selection & Snippet Compatibility ─────────
    @patch("src.ingestion.youtube._get_transcript_list")
    def test_handles_fetched_transcript_snippet_objects(self, mock_get_list):
        """youtube-transcript-api >=1.0 returns FetchedTranscriptSnippet dataclass objects without .get()"""
        class DummySnippet:
            def __init__(self, text, start, duration):
                self.text = text
                self.start = start
                self.duration = duration
                # Explicitly no .get() method to replicate FetchedTranscriptSnippet

        mock_manual = MagicMock()
        mock_manual.is_generated = False
        mock_manual.language_code = "en"
        mock_manual.fetch.return_value = [
            DummySnippet("First snippet text", 0.0, 5.0),
            DummySnippet("Second snippet text", 5.0, 4.0),
        ]

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.return_value = mock_manual
        mock_get_list.return_value = mock_transcript_list

        segments, origin, lang = fetch_transcript("sample_video_id")
        self.assertEqual(len(segments), 2)
        self.assertIsInstance(segments[0], dict)
        self.assertEqual(segments[0]["text"], "First snippet text")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["duration"], 5.0)

    @patch("src.ingestion.youtube._get_transcript_list")
    def test_prefers_creator_over_auto_captions(self, mock_get_list):
        mock_manual = MagicMock()
        mock_manual.is_generated = False
        mock_manual.language_code = "en"
        mock_manual.fetch.return_value = [{"text": "Manual subtitle text", "start": 0.0, "duration": 2.0}]

        mock_auto = MagicMock()
        mock_auto.is_generated = True
        mock_auto.language_code = "en"
        mock_auto.fetch.return_value = [{"text": "Auto subtitle text", "start": 0.0, "duration": 2.0}]

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.return_value = mock_manual
        mock_transcript_list.find_generated_transcript.return_value = mock_auto
        mock_get_list.return_value = mock_transcript_list

        segments, origin, lang = fetch_transcript("sample_video_id")

        self.assertEqual(origin, "creator")
        self.assertEqual(lang, "en")
        self.assertEqual(segments[0]["text"], "Manual subtitle text")
        mock_manual.fetch.assert_called_once()
        mock_auto.fetch.assert_not_called()

    @patch("src.ingestion.youtube._get_transcript_list")
    def test_falls_back_to_auto_captions_when_no_creator_captions(self, mock_get_list):
        mock_auto = MagicMock()
        mock_auto.is_generated = True
        mock_auto.language_code = "en"
        mock_auto.fetch.return_value = [{"text": "Auto caption fallback", "start": 0.0, "duration": 2.0}]

        mock_transcript_list = MagicMock()
        # Manual raises NoTranscriptFound
        mock_transcript_list.find_manually_created_transcript.side_effect = NoTranscriptFound("sample", [], [])
        mock_transcript_list.__iter__.return_value = [mock_auto]
        mock_transcript_list.find_generated_transcript.return_value = mock_auto
        mock_get_list.return_value = mock_transcript_list

        segments, origin, lang = fetch_transcript("sample_video_id")

        self.assertEqual(origin, "auto")
        self.assertEqual(lang, "en")
        self.assertEqual(segments[0]["text"], "Auto caption fallback")

    # ── 5. Caption Unavailable / Error Handling ─────────────────────────────
    @patch("src.ingestion.youtube._get_transcript_list")
    def test_captions_disabled_error(self, mock_get_list):
        mock_get_list.side_effect = TranscriptsDisabled("sample_video_id")
        with self.assertRaises(DocumentProcessingError) as ctx:
            fetch_transcript("sample_video_id")
        self.assertIn("Captions are disabled", str(ctx.exception))

    @patch("src.ingestion.youtube._get_transcript_list")
    def test_video_unavailable_error(self, mock_get_list):
        mock_get_list.side_effect = VideoUnavailable("sample_video_id")
        with self.assertRaises(DocumentProcessingError) as ctx:
            fetch_transcript("sample_video_id")
        self.assertIn("unavailable", str(ctx.exception))

    @patch("src.ingestion.youtube._get_transcript_list")
    def test_ip_blocked_error(self, mock_get_list):
        mock_get_list.side_effect = IpBlocked("sample_video_id")
        with self.assertRaises(DocumentProcessingError) as ctx:
            fetch_transcript("sample_video_id")
        self.assertIn("blocked or rate-limited", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
