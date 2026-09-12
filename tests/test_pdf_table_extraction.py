"""Regression test for excluding detected table text from PDF prose."""

import sys
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FilteredPage:
    def extract_text(self):
        return "Caption before table\nDiscussion after table"


class _Page:
    def __init__(self):
        self.filter_called = False

    def find_tables(self):
        return [_Table()]

    def filter(self, predicate):
        self.filter_called = True
        # Verify that prose is retained and a character in the table box is
        # excluded before pdfplumber's filtered page extracts text.
        self.prose_kept = predicate({
            "object_type": "char", "x0": 5, "x1": 6, "top": 5, "bottom": 6,
        })
        self.table_removed = not predicate({
            "object_type": "char", "x0": 50, "x1": 51, "top": 50, "bottom": 51,
        })
        return _FilteredPage()

    def extract_text(self):
        return "Caption before table\nTABLE GLYPHS\nDiscussion after table"


class _Table:
    bbox = (40, 40, 100, 100)

    def extract(self):
        return [["Heading"], ["Table value"]]


class _Pdf:
    def __init__(self, page):
        self.pages = [page]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _load_processor_with_lightweight_stubs(page):
    fastapi = types.ModuleType("fastapi")
    fastapi.UploadFile = object
    sys.modules.setdefault("fastapi", fastapi)

    pdfplumber = types.ModuleType("pdfplumber")
    pdfplumber.open = lambda _stream: _Pdf(page)
    sys.modules["pdfplumber"] = pdfplumber

    splitters = types.ModuleType("langchain_text_splitters")
    splitters.RecursiveCharacterTextSplitter = object
    splitters.MarkdownHeaderTextSplitter = object
    sys.modules.setdefault("langchain_text_splitters", splitters)

    config = types.ModuleType("src.core.config")
    config.settings = types.SimpleNamespace()
    config.CHUNK_SIZE_PRESETS = {"default": 2000}
    config.DocType = str
    sys.modules["src.core.config"] = config

    headers = types.ModuleType("src.core.header_detector")
    headers.HeaderDetector = object
    headers.NormalizationResult = object
    sys.modules["src.core.header_detector"] = headers

    blocks = types.ModuleType("src.core.block_extractor")
    blocks.BlockExtractor = object
    blocks.ExtractionResult = object
    blocks.ExtractedBlock = object
    sys.modules["src.core.block_extractor"] = blocks

    errors = types.ModuleType("src.utils.errors")
    errors.DocumentProcessingError = RuntimeError
    sys.modules["src.utils.errors"] = errors

    sys.modules.pop("src.ingestion.processor", None)
    from src.ingestion.processor import DocumentProcessor
    return DocumentProcessor


class PdfTableExtractionTests(unittest.TestCase):
    def test_tables_are_excluded_from_prose_and_emitted_once_as_markdown(self):
        page = _Page()
        processor = _load_processor_with_lightweight_stubs(page)

        extracted = processor._extract_pdf(b"pdf", "example.pdf")

        self.assertTrue(page.filter_called)
        self.assertTrue(page.prose_kept)
        self.assertTrue(page.table_removed)
        self.assertNotIn("TABLE GLYPHS", extracted)
        self.assertEqual(extracted.count("| Heading |"), 1)
        self.assertIn("Caption before table", extracted)
        self.assertIn("Discussion after table", extracted)


if __name__ == "__main__":
    unittest.main()
