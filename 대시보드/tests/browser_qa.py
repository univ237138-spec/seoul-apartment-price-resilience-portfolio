"""Local browser QA. Uses an isolated headless Chrome session, no user profile."""
import json
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if (APP_DIR / ".test_deps").exists():
    sys.path.insert(0, str(APP_DIR / ".test_deps"))
from playwright.sync_api import sync_playwright


def run():
    output = APP_DIR / ".qa"
    output.mkdir(exist_ok=True)
    results = {"errors": [], "screenshots": [], "overflow": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: results["errors"].append(str(error)))

        def capture(name):
            size = page.viewport_size
            main = page.get_by_test_id("stMain")
            total = main.evaluate("el => el.scrollHeight")
            page.set_viewport_size({"width": size["width"], "height": min(14000, total + 80)})
            main.evaluate("el => el.scrollTop = 0")
            page.get_by_role("combobox", name="자치구", exact=True).wait_for()
            page.get_by_role("button", name="결과 보기", exact=True).wait_for()
            page.locator(".js-plotly-plot .main-svg").first.wait_for()
            page.wait_for_load_state("networkidle", timeout=20000)
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            page.screenshot(path=str(output / f"{name}.png"), full_page=True)
            results["screenshots"].append(f"{name}.png")
            page.set_viewport_size(size)
            main.evaluate("el => el.scrollTop = 0")
            results["overflow"][name] = page.evaluate("document.documentElement.scrollWidth > innerWidth") or main.evaluate("el => el.scrollWidth > el.clientWidth")

        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.get_by_text("아직은 가격 회복력을", exact=False).wait_for(timeout=60000)
        page.get_by_text("1,388만원/㎡", exact=True).wait_for(timeout=60000)
        page.get_by_text("우리집 가격체력", exact=True).last.wait_for()
        capture("real-desktop")
        page.get_by_test_id("stRadioOption").filter(has_text="시연 화면").click()
        page.get_by_text("4,118만원/㎡", exact=True).wait_for(timeout=60000)
        page.get_by_text("80점", exact=True).wait_for()
        capture("demo-desktop")
        # Verify keyboard-operable selectors and atomic apply in a real browser.
        apartment = page.get_by_test_id("stSelectbox").nth(2)
        apartment.click()
        page.get_by_role("option", name="문정시영 · 문정동", exact=True).click()
        page.get_by_text("변경한 조건은", exact=False).wait_for()
        assert page.get_by_text("4,118만원/㎡", exact=True).count() == 1
        page.get_by_role("button", name="결과 보기", exact=True).click()
        page.get_by_text("1,650만원/㎡", exact=True).wait_for()
        page.get_by_text("65점", exact=True).wait_for()
        apartment.click()
        page.get_by_role("option", name="리센츠 · 잠실동", exact=True).click()
        page.get_by_role("button", name="결과 보기", exact=True).click()
        page.get_by_text("80점", exact=True).wait_for()
        page.set_viewport_size({"width": 390, "height": 844})
        capture("demo-mobile")
        page.screenshot(path=str(output / "demo-mobile-top.png"))
        page.get_by_test_id("stRadioOption").filter(has_text="실데이터").click()
        page.get_by_text("1,388만원/㎡", exact=True).wait_for()
        assert page.get_by_text("80점", exact=True).count() == 0
        capture("real-mobile")
        browser.close()
    (output / "browser-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    assert not results["errors"], results["errors"]
    assert not any(results["overflow"].values()), results["overflow"]


if __name__ == "__main__":
    run()
