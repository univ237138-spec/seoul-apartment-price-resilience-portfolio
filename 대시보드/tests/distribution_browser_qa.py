"""Read-only UI checks on an isolated browser against a local running dashboard."""
import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright


def run(url):
    output = Path(__file__).resolve().parents[1] / ".qa"
    output.mkdir(exist_ok=True)
    results = {"errors": [], "screenshots": [], "checks": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: results["errors"].append(str(error)))
        page.goto(url, wait_until="domcontentloaded")
        trend_tab = page.get_by_role("tab", name="가격 추이", exact=True)
        distribution_tab = page.get_by_role("tab", name="단지 분포", exact=True)
        trend_tab.wait_for(timeout=180000)
        trend = page.locator('.st-key-price-chart-card')
        scatter = page.locator('.st-key-distribution-chart-card')
        assert trend.is_visible() and not scatter.is_visible()
        trend.locator('.js-plotly-plot').wait_for()
        trend.locator('input[type="checkbox"]').check()
        page.wait_for_function("document.querySelector('.st-key-price-chart-card .js-plotly-plot')?.layout.yaxis.range[0] === 0")
        distribution_tab.click()
        plot = scatter.locator('.js-plotly-plot')
        plot.wait_for()
        assert not trend.is_visible()
        results["real_summary"] = page.locator('.distribution-insights').inner_text()
        results["plot"] = plot.evaluate("el => el.data.filter(t => t.name).map(t => ({name:t.name, count:t.x.length, selected:t.name === '선택 단지' ? t.customdata : null}))")
        selected_trace = next(t for t in results["plot"] if t["name"] == "선택 단지")
        assert selected_trace["count"] == 1
        assert "리센츠" in selected_trace["selected"][0][0]
        for width, label in [(1440, "desktop"), (900, "tablet"), (390, "mobile")]:
            page.set_viewport_size({"width": width, "height": 1100})
            plot.scroll_into_view_if_needed()
            page.wait_for_function("Array.from(document.querySelectorAll('.st-key-distribution-chart-card .main-svg')).some(e => e.getBoundingClientRect().width > 0)")
            assert not page.get_by_test_id("stException").count()
            assert page.get_by_test_id("stMain").evaluate("el => el.scrollWidth <= el.clientWidth"), label
            assert scatter.evaluate("el => el.scrollWidth <= el.clientWidth"), label
            scatter.screenshot(path=str(output / f"distribution-{label}.png"))
            results["screenshots"].append(f"distribution-{label}.png")
        page.set_viewport_size({"width": 1440, "height": 1100})
        trend_tab.click()
        assert trend.is_visible() and not scatter.is_visible()
        assert trend.locator('input[type="checkbox"]').is_checked()
        trend.screenshot(path=str(output / "distribution-trend-tab.png"))
        results["checks"].append("tab switch and zero-axis toggle preserved")

        # Draft filters must not replace the applied distribution until the user applies them.
        page.get_by_role("combobox", name="자치구", exact=True).click()
        page.get_by_role("option", name="강남구", exact=True).click()
        page.get_by_text("변경한 조건은", exact=False).wait_for()
        assert "송파구" in page.locator('.distribution-heading').inner_text()
        page.get_by_role("button", name="결과 보기", exact=True).click()
        page.wait_for_function("document.querySelector('.distribution-heading')?.textContent.includes('강남구')")
        distribution_tab.click()
        assert "강남구" in page.locator('.distribution-heading').inner_text()
        results["checks"].append("district changes only on apply")

        page.get_by_role("combobox", name="전용면적", exact=True).click()
        page.get_by_role("option", name="85㎡ 이상 102㎡ 미만", exact=True).click()
        page.get_by_text("변경한 조건은", exact=False).wait_for()
        assert "60㎡ 이상 85㎡ 미만" in page.locator('.distribution-heading').inner_text()
        page.get_by_role("button", name="결과 보기", exact=True).click()
        page.wait_for_function("document.querySelector('.distribution-heading')?.textContent.includes('85㎡ 이상 102㎡ 미만')")
        results["checks"].append("area changes only on apply")
        assert not page.get_by_test_id("stException").count()
        page.get_by_test_id("stRadioOption").filter(has_text="시연 화면").click()
        page.get_by_text("시연용 가격 흐름으로 계산한 예시 분포예요.", exact=False).wait_for(state="attached")
        distribution_tab.click()
        plot.wait_for()
        page.wait_for_function("document.querySelector('.st-key-distribution-chart-card .js-plotly-plot')?.data.filter(t => t.name).reduce((n,t) => n+t.x.length,0) === 4")
        results["checks"].append("demo has exactly 4 example complexes")
        assert not results["errors"], results["errors"]
        browser.close()
    (output / "distribution-browser-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8517")
