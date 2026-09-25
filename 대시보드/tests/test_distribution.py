from dataclasses import replace
import json
import unittest

import pandas as pd

from test_dashboard import fixture
import test_dashboard as dashboard_tests
from housing.distribution import distribution_values, distribution_narrative
from housing.distribution_ui import distribution_chart
from housing.metrics import build_snapshot
from housing.models import AREA_LABELS

AREA = AREA_LABELS[2]


def population():
    data = fixture([100, 100, 80, 90, 110])
    shapes = {"unit": [100, 100, 80, 90, 110], "peer": [100, 100, 60, 90, 90],
              "peer2": [100, 100, 70, 90, 120], "rising": [100, 110, 120, 130, 140],
              "short": [100, 90, 110], "other-district": [100, 100, 95, 100, 200]}
    frames, records = [], []
    for cid, prices in shapes.items():
        frame = fixture(prices).quarterly.copy()
        frame["complex_id"] = cid
        frames.append(frame)
        records.append({"complex_id": cid, "name": cid, "district": "강남구" if cid == "other-district" else "송파구",
                        "dong": "테스트동", "address": f"주소 {cid}"})
    other_area = frames[0].copy()
    other_area["area_group"] = AREA_LABELS[3]
    other_area["price"] = [100, 100, 50, 100, 300]
    frames.append(other_area)
    return replace(data, quarterly=pd.concat(frames, ignore_index=True), complexes=pd.DataFrame(records))


class DistributionTests(unittest.TestCase):
    def view(self, data=None, cid="unit", area=AREA):
        data = data if data is not None else population()
        return distribution_values(data, build_snapshot(data, cid, area))

    def test_exact_district_area_population_and_exclusions(self):
        view = self.view()
        self.assertEqual(set(view["points"].complex_id), {"unit", "peer", "peer2"})
        self.assertEqual((view["candidates"], view["insufficient"], view["no_decline"]), (5, 1, 1))
        self.assertAlmostEqual(view["selected"]["mdd"], -20)
        self.assertAlmostEqual(view["selected"]["gap"], 10)
        self.assertAlmostEqual(view["median_mdd"], -35)
        self.assertAlmostEqual(view["median_gap"], 5)
        self.assertEqual(view["other_count"], 2)
        headline, explanation = distribution_narrative(view)
        self.assertIn("덜 떨어졌고", headline)
        self.assertIn("더 높아요", headline)
        self.assertIn("비교 단지가 적어", explanation)

    def test_every_point_matches_existing_snapshot_even_with_thin_latest_quarter(self):
        data = population()
        latest = data.quarterly.quarter.eq(data.reference)
        data.quarterly.loc[latest, "complete"] = False
        data.quarterly.loc[latest & data.quarterly.complex_id.eq("peer"), "count"] = 1
        data.partial = True
        data.quarterly = data.quarterly.sample(frac=1, random_state=17)
        view = self.view(data)
        for row in view["points"].itertuples():
            snapshot = build_snapshot(data, row.complex_id, AREA)
            self.assertAlmostEqual(row.mdd, snapshot.metrics["mdd"].value)
            self.assertAlmostEqual(row.gap, snapshot.metrics["gap"].value)
            self.assertEqual(row.basis, snapshot.basis)
            self.assertEqual(row.count, snapshot.metrics["quarter_count"].value)
        self.assertEqual(view["stale_count"], 1)

    def test_unplottable_selected_does_not_get_a_fake_zero(self):
        for cid, reason in [("short", "기록이 부족"), ("rising", "하락 구간이 없어")]:
            view = self.view(cid=cid)
            self.assertIsNone(view["selected"])
            self.assertNotIn(cid, view["points"].complex_id.tolist())
            self.assertIn(reason, distribution_narrative(view)[1])
            self.assertNotIn("선택 단지", [trace.name for trace in distribution_chart(view).data])

    def test_empty_and_single_point_never_claim_relative_superiority(self):
        data = population()
        data.quarterly = data.quarterly.loc[data.quarterly.complex_id.eq("short")]
        view = self.view(data, "short")
        self.assertTrue(view["points"].empty)
        self.assertEqual(len(distribution_chart(view).data), 0)
        only = self.view(area=AREA_LABELS[3])
        self.assertEqual(len(only["points"]), 1)
        self.assertIsNone(only["median_mdd"])
        self.assertIn("비교할 다른 단지가 없어요", distribution_narrative(only)[0])

    def test_all_outliers_are_included_and_selected_is_highlighted(self):
        data = population()
        data.quarterly.loc[data.quarterly.complex_id.eq("peer2") & data.quarterly.quarter.eq(data.reference), "price"] = 2000
        view = self.view(data)
        chart = distribution_chart(view)
        traces = {trace.name: trace for trace in chart.data if trace.name}
        self.assertEqual(len(traces["비교 단지"].x), 2)
        self.assertEqual(len(traces["선택 단지"].x), 1)
        self.assertEqual(traces["선택 단지"].customdata[0][-1], "unit")
        self.assertGreater(chart.layout.yaxis.range[1], 1900)
        self.assertLess(chart.layout.yaxis.range[0], 0)


class DistributionAppTests(unittest.TestCase):
    setUp = dashboard_tests.AppTests.setUp
    app = dashboard_tests.AppTests.app
    def test_tabs_follow_applied_filters_and_keep_trend_toggle(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual([tab.label for tab in app.tabs], ["가격 추이", "단지 분포"])
        self.assertIn("송파구", "".join(el.value for el in app.markdown if "distribution-heading" in el.value))
        app.selectbox(key="real_district").select("강남구").run()
        self.assertIn("송파구", "".join(el.value for el in app.markdown if "distribution-heading" in el.value))
        app.button[0].click().run()
        self.assertIn("강남구", "".join(el.value for el in app.markdown if "distribution-heading" in el.value))
        self.assertFalse(app.exception)
        app.toggle[0].set_value(True).run()
        self.assertFalse(app.exception)
        chart = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(chart["layout"]["yaxis"]["range"][0], 0)


if __name__ == "__main__":
    unittest.main()
