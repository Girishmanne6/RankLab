"""Capture RankLab dashboard screenshots for the README."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
BASE = "http://127.0.0.1:5173"


def shot(page, name: str) -> None:
    path = OUT / name
    page.wait_for_timeout(1200)
    page.screenshot(path=str(path), full_page=True)
    print(f"wrote {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(BASE, wait_until="networkidle")

        # Tab 1: Model Comparison (default)
        page.get_by_role("button", name="Model Comparison").click()
        shot(page, "model-comparison.png")

        # Tab 2: Fairness Analysis
        page.get_by_role("button", name="Fairness Analysis").click()
        shot(page, "fairness-analysis.png")

        # Tab 3: Recommendations
        page.get_by_role("button", name="Recommendations").click()
        sample = page.locator("button.tab").filter(has_text="0").first
        if sample.count():
            sample.click()
        else:
            page.get_by_placeholder("User ID (e.g. 42)").fill("0")
            page.get_by_role("button", name="Recommend").click()
        shot(page, "recommendations.png")

        browser.close()


if __name__ == "__main__":
    main()
