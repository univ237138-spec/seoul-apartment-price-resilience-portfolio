from dataclasses import replace
import unittest
from unittest.mock import patch

import pandas as pd

from test_dashboard import fixture, raw_row
from housing.data import prepare_data
from housing.metrics import build_snapshot
from housing.models import AREA_LABELS
from housing.price_overview import overview_values
from housing.price_overview_ui import amount, render_price_overview
from housing.visuals import price_chart


def partial_data(last=130, count=5):
    data = fixture([100, 120, 80, 100, last], [3, 3, 3, 3, count])
    data.quarterly.loc[4, "complete"] = False
    return replace(data, partial=True, end=pd.Timestamp("2021-02-28"))


class PriceOverviewTests(unittest.TestCase):
    def test_partial_price_recovery_and_card_consistency(self):
        data = partial_data()
        s = build_snapshot(data, "unit", AREA_LABELS[2])
        view = overview_values(data, s)
        self.assertEqual(s.metrics["price"].status, "partial")
        self.assertAlmostEqual(view["mdd"], -100 / 3)
        self.assertAlmostEqual(view["gap"], 100 / 12)
        self.assertEqual(view["price"], 130)
        self.assertEqual(amount(view["price"], view["area"]), "1.1억원")
        self.assertEqual(view["count"], 5)
        self.assertEqual((view["qualified"], view["total_quarters"]), (5, 5))
        self.assertEqual(view["current_period"], "2021년 1~2월")
        self.assertEqual(view["decline_duration"], "3개월")
        self.assertEqual(view["since_peak"], "9개월")
        self.assertEqual(view["recovery_duration"], "6개월 만에 첫 회복 · 잠정")
        notes = price_chart(s).data[0].customdata
        self.assertNotIn("잠정값", notes[0][2])
        self.assertIn("잠정값", notes[-1][2])

    def test_latest_partial_new_low_is_part_of_mdd(self):
        s = build_snapshot(partial_data(last=60), "unit", AREA_LABELS[2])
        self.assertEqual(s.metrics["mdd"].value, -50)
        self.assertEqual(s.metrics["trough"].period, "2021Q1")
        self.assertEqual(s.metrics["recovery_quarters"].status, "unrecovered")

    def test_thin_latest_quarter_is_reported_as_a_stepped_back_basis(self):
        # 최신 분기가 2건뿐이면 그 값을 "현재"로 내세우지 않고, 3건 이상이던 직전 분기로 물러난다.
        data = partial_data(count=2)
        s = build_snapshot(data, "unit", AREA_LABELS[2])
        self.assertEqual(s.basis, pd.Period("2020Q4"))
        self.assertEqual(s.metrics["price"].value, 100)
        self.assertIsNotNone(s.metrics["mdd"].value)
        view = overview_values(data, s)
        self.assertTrue(view["stale"])
        self.assertEqual(view["current_label"], "마지막 확인")
        self.assertEqual(view["latest_period"], "2021년 1~2월")
        # 관측 자체가 없는 최신 분기는 여전히 빈칸으로 남는다.
        missing = replace(data, quarterly=data.quarterly.iloc[:4].copy())
        s = build_snapshot(missing, "unit", AREA_LABELS[2])
        self.assertTrue(pd.isna(s.series.price.iloc[-1]))

    def test_open_area_bin_uses_trade_median_not_quarter_medians(self):
        rows = [raw_row(**{"전용면적(㎡)": str(area), "계약년월": month}) for area, month in
                [(135, "2020-01"), (135, "2020-01"), (135, "2020-01"), (200, "2020-04"), (80, "2020-04")]]
        data = prepare_data(pd.DataFrame(rows))
        cid = data.complexes.complex_id.iloc[0]
        large = overview_values(data, build_snapshot(data, cid, AREA_LABELS[5]))
        regular = overview_values(data, build_snapshot(data, cid, AREA_LABELS[2]))
        self.assertEqual(large["area"], 135)
        self.assertEqual(regular["area"], 85)

    def test_recent_year_count_survives_the_partial_reference_quarter(self):
        # The latest-quarter policy makes every real dataset partial, so gating this on a finished
        # reference quarter blanked "최근 1년" on every card.
        s = build_snapshot(partial_data(), "unit", AREA_LABELS[2])
        self.assertEqual(s.metrics["year_count"].value, 14)  # 2020Q2~2021Q1: 3+3+3+5
        self.assertEqual(s.metrics["year_count"].status, "partial")
        short = fixture([100, 120], [3, 3])
        self.assertIsNone(build_snapshot(short, "unit", AREA_LABELS[2]).metrics["year_count"].value)

    def test_no_decline_is_not_division_by_zero(self):
        data = partial_data(last=150)
        data.quarterly["price"] = [100, 110, 120, 130, 150]
        view = overview_values(data, build_snapshot(data, "unit", AREA_LABELS[2]))
        self.assertEqual(view["mdd"], 0)
        self.assertIsNone(view["gap"])
        self.assertEqual(view["status"], "하락 구간 없음")

    def test_price_cards_name_mdd_and_peak_position_without_recovery_rate_card(self):
        data = partial_data()
        view = overview_values(data, build_snapshot(data, "unit", AREA_LABELS[2]))
        with patch("housing.price_overview_ui.st.markdown") as markdown:
            render_price_overview(view)
        markup = markdown.call_args.args[0]
        self.assertIn("최대 낙폭(MDD)", markup)
        self.assertIn("고점 대비 위치", markup)
        self.assertNotIn("하락분 회복률", markup)


if __name__ == "__main__":
    unittest.main()
