"""Score disclosure interaction and responsive QA against a running local dashboard.

Run: python tests/score_browser_qa.py http://127.0.0.1:8517
Uses an isolated headless browser; writes screenshots only under .qa/.
"""
import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


def run(url):
    output = Path(__file__).resolve().parents[1] / ".qa"
    output.mkdir(exist_ok=True)
    results = {"errors": [], "screenshots": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: results["errors"].append(str(error)))
        page.goto(url, wait_until="domcontentloaded")
        toggle = page.locator('.score-details > summary')
        toggle.wait_for(timeout=180000)
        assert page.get_by_test_id("stException").count() == 0
        popup = page.locator('.score-popover')
        assert not popup.is_visible()
        toggle.click()
        popup.wait_for(state="visible")
        assert page.locator('.score-axis-track').count() == 2
        assert page.locator('.grade-band.selected').count() == 1
        results["real_score_text"] = popup.inner_text()

        for width, height, label in [(1440, 1100, "desktop"), (768, 1100, "tablet"), (390, 1000, "mobile"), (320, 1000, "narrow")]:
            page.set_viewport_size({"width": width, "height": height})
            toggle.scroll_into_view_if_needed()
            box = popup.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= width, box
            assert page.get_by_test_id("stMain").evaluate("el => el.scrollWidth <= el.clientWidth"), label
            popup.screenshot(path=str(output / f"score-{label}.png"))
            results["screenshots"].append(f"score-{label}.png")
            # The overlay must remain above the following price cards and chart.
            assert popup.evaluate("el => { const r = el.getBoundingClientRect(); const x = r.x + r.width / 2; const y = Math.min(innerHeight - 2, r.bottom - 8); return y < r.top || el.contains(document.elementFromPoint(x, y)); }"), label

        page.set_viewport_size({"width": 1440, "height": 1100})
        toggle.focus()
        page.keyboard.press("Enter")
        assert not popup.is_visible()
        page.keyboard.press("Space")
        assert popup.is_visible()
        page.locator('.score-method-notes > summary').click()
        assert page.locator('.score-method-notes').get_attribute("open") is not None
        toggle.click()
        page.get_by_test_id("stRadioOption").filter(has_text="시연 화면").click()
        page.locator('.source-strip.demo').wait_for()
        page.locator('.score-popover h3').filter(has_text="시연용 점수").wait_for(state="attached")
        toggle.click()
        assert "시연용 점수" in popup.inner_text()
        assert page.locator('.grade-scale').count() == 0
        assert not results["errors"], results["errors"]
        browser.close()
    (output / "score-browser-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8517")
