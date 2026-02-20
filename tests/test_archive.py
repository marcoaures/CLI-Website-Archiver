import sys
import types
import unittest

import json
import os
import tempfile

# Provide lightweight stubs so helper-function tests can import archive.py
requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

lxml_stub = types.ModuleType("lxml")
lxml_stub.etree = types.SimpleNamespace()
sys.modules.setdefault("lxml", lxml_stub)

pil_stub = types.ModuleType("PIL")
pil_stub.Image = types.SimpleNamespace()
sys.modules.setdefault("PIL", pil_stub)

playwright_stub = types.ModuleType("playwright")
playwright_async_stub = types.ModuleType("playwright.async_api")
playwright_async_stub.TimeoutError = Exception
playwright_async_stub.async_playwright = lambda: None
sys.modules.setdefault("playwright", playwright_stub)
sys.modules.setdefault("playwright.async_api", playwright_async_stub)

tqdm_stub = types.ModuleType("tqdm")
tqdm_stub.tqdm = lambda *args, **kwargs: None
sys.modules.setdefault("tqdm", tqdm_stub)

import archive


class ArchiveHelpersTest(unittest.TestCase):
    def test_normalize_domain_to_base_url(self):
        self.assertEqual(archive.normalize_domain_to_base_url("example.com"), "https://example.com")
        self.assertEqual(archive.normalize_domain_to_base_url("https://example.com/path?q=1"), "https://example.com")

    def test_should_skip_url_by_prefix(self):
        self.assertTrue(archive.should_skip_url_by_prefix("https://example.com/blog/post", ["/blog"]))
        self.assertFalse(archive.should_skip_url_by_prefix("https://example.com/docs/post", ["/blog"]))

    def test_safe_slug_from_url_contains_host(self):
        slug = archive.safe_slug_from_url("https://example.com/a/b?x=1")
        self.assertIn("example.com", slug)
        self.assertTrue(slug.endswith("x_1"))

    def test_load_config_defaults_consent_selectors_to_blank(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump({"out_dir": "archive_out", "sitemap_urls": []}, tmp)
            path = tmp.name

        try:
            cfg = archive.load_config(path)
            self.assertEqual(cfg.consent_accept_selector, "")
            self.assertEqual(cfg.consent_modal_selector, "")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
