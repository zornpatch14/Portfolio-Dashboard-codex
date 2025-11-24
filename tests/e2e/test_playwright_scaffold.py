"""Playwright scaffold to validate basic browser launch flows."""

from __future__ import annotations

import pytest

playwright_sync = pytest.importorskip(
    "playwright.sync_api", reason="Playwright not installed; skipping scaffold"
)


def test_playwright_browser_launch():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("about:blank")
        assert "about:blank" in page.url
        browser.close()
