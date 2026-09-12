"""Regression coverage for the default academic-heading policy."""

import unittest
from pathlib import Path
import sys
import types


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing ``src.core.header_detector`` initialises ``src.core`` first. The
# detector itself has no settings dependency, so a lightweight config module
# keeps this unit test independent of optional application dependencies.
config = types.ModuleType("src.core.config")
config.settings = types.SimpleNamespace()
sys.modules.setdefault("src.core.config", config)

from src.core.header_detector import HeaderDetector


class HeaderDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = HeaderDetector()

    def test_default_accepts_all_caps_academic_heading(self):
        body = "x" * 220
        result = self.detector.normalize(f"ABSTRACT\n{body}")

        self.assertIn("# ABSTRACT", result.normalized_text)
        self.assertTrue(result.detected_headers[0].accepted)

    def test_default_accepts_title_case_heading(self):
        result = self.detector.normalize("\nLiterature Review\nThis section surveys prior work.")

        self.assertIn("# Literature Review", result.normalized_text)
        self.assertTrue(result.detected_headers[0].accepted)

    def test_sentence_like_line_is_not_promoted_to_heading(self):
        text = "\nThis ordinary sentence has lowercase words\nIt remains prose."
        result = self.detector.normalize(text)

        self.assertNotIn("# This ordinary sentence", result.normalized_text)
        self.assertEqual(result.detected_headers, [])

    def test_numbered_heading_remains_accepted(self):
        result = self.detector.normalize("1. Introduction\nBody text.")

        self.assertIn("# 1. Introduction", result.normalized_text)


if __name__ == "__main__":
    unittest.main()
