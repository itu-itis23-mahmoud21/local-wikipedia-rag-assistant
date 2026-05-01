"""Tests for the Wikipedia API client and text helpers."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from src.wiki_client import (
    WikipediaClient,
    WikipediaClientError,
    WikipediaPageNotFoundError,
    build_wikipedia_page_url,
    normalize_wikipedia_text,
    read_text_file,
    remove_unwanted_wikipedia_sections,
    safe_filename,
    write_text_file,
)


class TestWikipediaHelpers(unittest.TestCase):
    """Tests for local filename and text helpers."""

    def test_safe_filename_converts_common_entity_names(self) -> None:
        """safe_filename should produce stable non-blank filename stems."""

        self.assertIn("albert_einstein", safe_filename("Albert Einstein"))
        self.assertEqual(safe_filename("J. K. Rowling"), "j_k_rowling")
        self.assertTrue(safe_filename("Sagrada Família"))

    def test_normalize_wikipedia_text_collapses_blank_lines(self) -> None:
        """Text normalization should strip and collapse excessive blank lines."""

        text = "  First line\r\n\r\n\r\n Second line \n\n\nThird line  "

        normalized = normalize_wikipedia_text(text)

        self.assertEqual(normalized, "First line\n\nSecond line\n\nThird line")
        self.assertNotIn("\r", normalized)

    def test_normalize_wikipedia_text_removes_references_section(self) -> None:
        """Standalone References should remove trailing footer content."""

        text = "Intro paragraph.\n\nReferences\nReference item"

        normalized = normalize_wikipedia_text(text)

        self.assertEqual(normalized, "Intro paragraph.")
        self.assertNotIn("Reference item", normalized)

    def test_normalize_wikipedia_text_removes_see_also_section(self) -> None:
        """Standalone See also should remove trailing footer content."""

        text = "Main article text.\n\nSee also\nRelated page"

        normalized = normalize_wikipedia_text(text)

        self.assertEqual(normalized, "Main article text.")
        self.assertNotIn("Related page", normalized)

    def test_normalize_wikipedia_text_removes_common_footer_headings(self) -> None:
        """Common standalone footer headings should be removed."""

        for heading in ["Works cited", "Further reading", "External links"]:
            with self.subTest(heading=heading):
                text = f"Useful article text.\n\n{heading}\nFooter content"

                normalized = normalize_wikipedia_text(text)

                self.assertEqual(normalized, "Useful article text.")
                self.assertNotIn("Footer content", normalized)

    def test_normalize_wikipedia_text_keeps_sentence_with_references_word(self) -> None:
        """Only standalone headings should trigger footer removal."""

        text = "The article references earlier experiments.\n\nSecond paragraph."

        normalized = normalize_wikipedia_text(text)

        self.assertIn("references earlier experiments", normalized)
        self.assertIn("Second paragraph.", normalized)

    def test_remove_unwanted_wikipedia_sections_is_case_insensitive(self) -> None:
        """Footer heading matching should ignore case."""

        text = "Article text.\n\nrEfErEnCeS\nFooter content"

        cleaned = remove_unwanted_wikipedia_sections(text)

        self.assertEqual(cleaned, "Article text.")

    def test_write_and_read_text_file_roundtrip_utf8(self) -> None:
        """Text helpers should roundtrip UTF-8 content."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "text.txt"
            text = "Beyoncé visited Sagrada Família."

            write_text_file(path, text)
            result = read_text_file(path)

        self.assertEqual(result, text)

    def test_build_wikipedia_page_url_for_albert_einstein(self) -> None:
        """Wikipedia URL builder should use underscores for spaces."""

        self.assertEqual(
            build_wikipedia_page_url("Albert Einstein"),
            "https://en.wikipedia.org/wiki/Albert_Einstein",
        )

    def test_build_wikipedia_page_url_for_j_k_rowling(self) -> None:
        """Wikipedia URL builder should preserve safe title punctuation."""

        url = build_wikipedia_page_url("J. K. Rowling")

        self.assertIn("J._K._Rowling", url)

    def test_build_wikipedia_page_url_for_unicode_title(self) -> None:
        """Wikipedia URL builder should encode non-ASCII titles."""

        url = build_wikipedia_page_url("Sagrada Fam\u00edlia")

        self.assertTrue(url)
        self.assertTrue(url.startswith("https://en.wikipedia.org/wiki/"))

    def test_build_wikipedia_page_url_rejects_blank_title(self) -> None:
        """Wikipedia URL builder should reject blank titles."""

        with self.assertRaises(ValueError):
            build_wikipedia_page_url("  ")


class TestWikipediaClient(unittest.TestCase):
    """Tests for WikipediaClient with mocked HTTP responses."""

    def setUp(self) -> None:
        """Create a client with deterministic settings."""

        self.client = WikipediaClient(
            base_url="https://example.test/api.php",
            timeout_seconds=5,
            user_agent="TestAgent/1.0",
        )

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_sends_expected_request_params(self, mock_get: Mock) -> None:
        """fetch_page should send the expected MediaWiki API parameters."""

        mock_get.return_value = _mock_response(
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 736,
                            "title": "Albert Einstein",
                            "fullurl": "https://en.wikipedia.org/wiki/Albert_Einstein",
                            "extract": "Albert Einstein was a physicist.",
                        }
                    ]
                }
            }
        )

        self.client.fetch_page("Albert Einstein")

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "https://example.test/api.php")

        params = kwargs["params"]
        self.assertEqual(params["action"], "query")
        self.assertEqual(params["prop"], "extracts|info")
        self.assertEqual(params["explaintext"], "1")
        self.assertEqual(params["exsectionformat"], "plain")
        self.assertEqual(params["redirects"], "1")
        self.assertEqual(params["inprop"], "url")
        self.assertEqual(params["titles"], "Albert Einstein")
        self.assertEqual(params["format"], "json")
        self.assertEqual(params["formatversion"], "2")
        self.assertEqual(params["utf8"], "1")
        self.assertEqual(kwargs["headers"]["User-Agent"], "TestAgent/1.0")
        self.assertEqual(kwargs["timeout"], 5)

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_success_parses_page_data(self, mock_get: Mock) -> None:
        """fetch_page should parse title, page id, URL, and extract."""

        mock_get.return_value = _mock_response(
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 736,
                            "title": "Albert Einstein",
                            "fullurl": "https://en.wikipedia.org/wiki/Albert_Einstein",
                            "extract": " Albert Einstein was a physicist. ",
                        }
                    ]
                }
            }
        )

        page = self.client.fetch_page("Albert Einstein")

        self.assertEqual(page.title, "Albert Einstein")
        self.assertEqual(page.page_id, 736)
        self.assertEqual(
            page.source_url,
            "https://en.wikipedia.org/wiki/Albert_Einstein",
        )
        self.assertEqual(page.extract, "Albert Einstein was a physicist.")
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["headers"]["User-Agent"], "TestAgent/1.0")
        self.assertEqual(kwargs["timeout"], 5)

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_handles_normalized_redirected_response(
        self,
        mock_get: Mock,
    ) -> None:
        """fetch_page should accept a normal page from redirected API output."""

        mock_get.return_value = _mock_response(
            {
                "query": {
                    "normalized": [
                        {"from": "albert einstein", "to": "Albert Einstein"}
                    ],
                    "redirects": [
                        {"from": "Einstein", "to": "Albert Einstein"}
                    ],
                    "pages": [
                        {
                            "pageid": 736,
                            "title": "Albert Einstein",
                            "fullurl": "https://en.wikipedia.org/wiki/Albert_Einstein",
                            "extract": "Redirected extract.",
                        }
                    ],
                }
            }
        )

        page = self.client.fetch_page("Einstein")

        self.assertEqual(page.title, "Albert Einstein")
        self.assertEqual(page.extract, "Redirected extract.")

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_raises_not_found_for_missing_page(
        self,
        mock_get: Mock,
    ) -> None:
        """Missing MediaWiki pages should raise WikipediaPageNotFoundError."""

        mock_get.return_value = _mock_response(
            {
                "query": {
                    "pages": [
                        {
                            "title": "Missing Page",
                            "missing": True,
                            "extract": "",
                        }
                    ]
                }
            }
        )

        with self.assertRaises(WikipediaPageNotFoundError):
            self.client.fetch_page("Missing Page")

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_raises_client_error_on_request_exception(
        self,
        mock_get: Mock,
    ) -> None:
        """Network exceptions should be wrapped in WikipediaClientError."""

        mock_get.side_effect = requests.RequestException("network down")

        with self.assertRaises(WikipediaClientError):
            self.client.fetch_page("Albert Einstein")

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_raises_client_error_on_non_200(
        self,
        mock_get: Mock,
    ) -> None:
        """Non-200 HTTP responses should raise WikipediaClientError."""

        mock_get.return_value = _mock_response({}, status_code=500, text="Server error")

        with self.assertRaises(WikipediaClientError):
            self.client.fetch_page("Albert Einstein")

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_raises_client_error_on_invalid_json_shape(
        self,
        mock_get: Mock,
    ) -> None:
        """Unexpected JSON response shapes should raise WikipediaClientError."""

        mock_get.return_value = _mock_response({"unexpected": {}})

        with self.assertRaises(WikipediaClientError):
            self.client.fetch_page("Albert Einstein")

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_raises_client_error_on_api_error_payload(
        self,
        mock_get: Mock,
    ) -> None:
        """Top-level MediaWiki API errors should include code and info."""

        mock_get.return_value = _mock_response(
            {
                "error": {
                    "code": "badmodule",
                    "info": "The module is invalid.",
                }
            }
        )

        with self.assertRaises(WikipediaClientError) as context:
            self.client.fetch_page("Albert Einstein")

        error_message = str(context.exception)
        self.assertIn("badmodule", error_message)
        self.assertIn("The module is invalid.", error_message)

    @patch("src.wiki_client.requests.get")
    def test_fetch_page_raises_client_error_on_invalid_json_body(
        self,
        mock_get: Mock,
    ) -> None:
        """Invalid JSON bodies should raise WikipediaClientError."""

        response = _mock_response({})
        response.json.side_effect = ValueError("bad json")
        mock_get.return_value = response

        with self.assertRaises(WikipediaClientError):
            self.client.fetch_page("Albert Einstein")

    def test_blank_title_raises_value_error(self) -> None:
        """Blank titles should be rejected before an HTTP request."""

        with self.assertRaises(ValueError):
            self.client.fetch_page("   ")


def _mock_response(
    payload: dict,
    status_code: int = 200,
    text: str = "OK",
) -> Mock:
    """Build a minimal mocked requests response."""

    response = Mock()
    response.status_code = status_code
    response.text = text
    response.json.return_value = payload
    return response
