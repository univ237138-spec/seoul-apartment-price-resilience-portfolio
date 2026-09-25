"""Acceptance tests; runnable with either unittest or pytest."""
from hashlib import sha256
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import unittest
from uuid import uuid4
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "sample_transactions.csv"
sys.path.insert(0, str(APP_DIR))
from housing.data import DEFAULT_CSV, DataError, area_groups, load_csv, prepare_data
from housing.demo import demo_dataset
from housing.metrics import build_snapshot
from housing.models import AREA_LABELS, Dataset
from housing.ui import comparison_details, josa_i_ga
from housing.visuals import facility_map, price_chart, radius_polygon


def fixture(prices, counts=None, quarters=None, reference=None):
    quarters = pd.PeriodIndex(quarters, freq="Q") if quarters else pd.period_range("2020Q1", periods=len(prices), freq="Q")
    frame = pd.DataFrame({"complex_id": "unit", "area_group": AREA_LABELS[2], "quarter": quarters,
                          "price": prices, "count": counts or [3] * len(prices), "complete": True})
    info = pd.DataFrame([{"complex_id": "unit", "name": "테스트", "dong": "테스트동", "latitude": None, "longitude": None}])
    ref = pd.Period(reference, freq="Q") if reference else quarters[-1]
    return Dataset("real", frame, info, pd.DataFrame(columns=["complex_id"]), quarters[0].start_time,
                   ref.end_time.normalize(), ref, False)


def raw_row(**changes):
    row = {"시군구": "서울특별시 송파구 잠실동", "단지명": "테스트", "도로명주소": "주소 1",
           "전용면적(㎡)": "84.99", "거래금액(만원)": "100,000", "계약년월": "2020-01", "계약일": "1"}
    row.update(changes)
    return row


class CalculationTests(unittest.TestCase):
    def test_area_boundaries(self):
        values = pd.Series([0, 39.99, 40, 59.99, 60, 84.99, 85, 101.99, 102, 134.99, 135])
        groups = area_groups(values).astype(str).tolist()
        self.assertEqual(groups, [AREA_LABELS[i] for i in [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]])

    def test_chronological_mdd_and_recovery_over_100(self):
        s = build_snapshot(fixture([100, 120, 80, 130]), "unit", AREA_LABELS[2])
        self.assertAlmostEqual(s.metrics["mdd"].value, -100 / 3)
        self.assertEqual(s.metrics["peak"].period, "2020Q2")
        self.assertEqual(s.metrics["trough"].period, "2020Q3")
        self.assertEqual(s.metrics["recovery"].value, 125)
        self.assertAlmostEqual(s.metrics["gap"].value, 100 / 12)
        self.assertEqual(s.metrics["recovery_quarters"].value, 1)

    def test_multiple_drawdowns_and_first_tie(self):
        s = build_snapshot(fixture([100, 50, 100, 50, 120]), "unit", AREA_LABELS[2])
        self.assertEqual(s.metrics["trough"].period, "2020Q2")
        self.assertEqual(s.metrics["mdd"].value, -50)
        self.assertEqual(s.metrics["recovery_quarters"].value, 1)

    def test_new_high_does_not_replace_preceding_peak(self):
        s = build_snapshot(fixture([100, 70, 110, 60, 120]), "unit", AREA_LABELS[2])
        self.assertEqual(s.metrics["peak"].value, 110)
        self.assertEqual(s.metrics["trough"].value, 60)
        self.assertAlmostEqual(s.metrics["mdd"].value, (60 / 110 - 1) * 100)

    def test_no_decline_and_zero_denominator(self):
        for prices in [[100] * 4, [100, 110, 120, 130]]:
            s = build_snapshot(fixture(prices), "unit", AREA_LABELS[2])
            self.assertEqual(s.metrics["mdd"].value, 0)
            self.assertIsNone(s.metrics["recovery"].value)
            self.assertEqual(s.metrics["recovery"].status, "no_decline")

    def test_not_recovered(self):
        s = build_snapshot(fixture([100, 60, 70, 80]), "unit", AREA_LABELS[2])
        self.assertEqual(s.metrics["recovery"].value, 50)
        self.assertEqual(s.metrics["recovery_quarters"].status, "unrecovered")

    def test_insufficient_counts_or_quarters(self):
        for data in [fixture([100, 80, 90]), fixture([100, 80, 90, 110], [3, 2, 3, 3])]:
            s = build_snapshot(data, "unit", AREA_LABELS[2])
            self.assertIsNone(s.metrics["mdd"].value)

    def test_thin_current_quarter_steps_back_to_a_named_basis(self):
        # 최신 분기가 얕으면 기준을 뒤로 물린다. 값을 옮겨오는 대신 실제로 관측된 분기를 쓰고,
        # 그게 어느 분기인지 snapshot이 들고 있어야 화면이 "현재"라고 잘못 말하지 않는다.
        s = build_snapshot(fixture([100, 80, 90, 110], reference="2021Q1"), "unit", AREA_LABELS[2])
        self.assertEqual(s.reference, pd.Period("2021Q1"))
        self.assertEqual(s.basis, pd.Period("2020Q4"))
        self.assertEqual(s.metrics["price"].value, 110)
        self.assertEqual(s.metrics["price"].period, "2020Q4")
        self.assertEqual(s.metrics["quarter_count"].value, 3)
        self.assertIsNotNone(s.metrics["mdd"].value)

    def test_no_qualifying_quarter_leaves_the_price_empty(self):
        # 3건 이상인 분기가 하나도 없으면 물러설 곳도 없다. 이때는 값을 지어내지 않는다.
        s = build_snapshot(fixture([100, 80, 90, 110], [2, 2, 2, 2]), "unit", AREA_LABELS[2])
        self.assertIsNone(s.metrics["price"].value)
        self.assertEqual(s.metrics["quarter_count"].value, 0)
        self.assertIsNone(s.metrics["mdd"].value)

    def test_missing_quarter_chart_not_interpolated(self):
        s = build_snapshot(fixture([100, 80, 90, 110], quarters=["2020Q1", "2020Q3", "2020Q4", "2021Q1"]), "unit", AREA_LABELS[2])
        self.assertTrue(pd.isna(s.series.loc[s.series.quarter == pd.Period("2020Q2"), "price"].iloc[0]))
        self.assertFalse(price_chart(s).data[0].connectgaps)

    def test_price_chart_discloses_zoomed_axis_and_can_start_at_zero(self):
        s = build_snapshot(fixture([1400, 1900, 1800, 2200]), "unit", AREA_LABELS[2])
        zoomed = price_chart(s)
        self.assertGreater(zoomed.layout.yaxis.range[0], 0)
        self.assertIn("// 0 생략", [note.text for note in zoomed.layout.annotations])
        from_zero = price_chart(s, zero_start=True)
        self.assertEqual(from_zero.layout.yaxis.range[0], 0)
        self.assertNotIn("0 생략", [note.text for note in (from_zero.layout.annotations or [])])


class DataTests(unittest.TestCase):
    def test_required_columns_and_empty(self):
        with self.assertRaisesRegex(DataError, "필수 컬럼"):
            prepare_data(pd.DataFrame({"wrong": [1]}))
        with self.assertRaisesRegex(DataError, "유효 거래"):
            prepare_data(pd.DataFrame([raw_row(**{"거래금액(만원)": "0"})]))

    def test_bad_values_cancelled_and_input_unchanged(self):
        rows = [raw_row(), raw_row(**{"해제사유발생일": "2020-01-02"}),
                raw_row(**{"계약일": "32"}), raw_row(**{"전용면적(㎡)": "0"}),
                raw_row(**{"거래금액(만원)": "inf"}), raw_row(**{"계약일": "1.5"})]
        raw = pd.DataFrame(rows)
        before = raw.copy(deep=True)
        data = prepare_data(raw)
        self.assertEqual(data.diagnostics["유효 거래"], 1)
        self.assertEqual(data.diagnostics["제외 거래"], 5)
        pd.testing.assert_frame_equal(raw, before)

    def test_median_of_ratios(self):
        rows = [raw_row(**{"거래금액(만원)": str(price), "전용면적(㎡)": str(area)})
                for price, area in [(6000, 60), (14000, 70), (8000, 80)]]
        data = prepare_data(pd.DataFrame(rows))
        self.assertEqual(data.quarterly.price.iloc[0], 100)

    def test_latest_quarter_partial_and_shared_reference(self):
        data = prepare_data(pd.DataFrame([raw_row(), raw_row(**{"계약년월": "2020-04", "계약일": "30"})]))
        self.assertEqual(data.reference, pd.Period("2020Q2"))
        self.assertTrue(data.partial)
        self.assertEqual(data.quarterly.complete.tolist(), [True, False])

    def test_names_do_not_merge_across_addresses(self):
        data = prepare_data(pd.DataFrame([raw_row(), raw_row(**{"도로명주소": "다른 주소"})]))
        self.assertEqual(len(data.complexes), 2)

    def test_only_normal_kapt_matches_use_code(self):
        data = prepare_data(pd.DataFrame([
            raw_row(**{"단지_KAPT단지코드": "A1", "단지매칭상태": "매칭완료"}),
            raw_row(**{"단지_KAPT단지코드": "A1", "단지매칭상태": "미매칭_후보충돌", "도로명주소": "주소 2"}),
        ]))
        self.assertEqual(len(data.complexes), 2)
        self.assertIn("kapt:A1", data.complexes.complex_id.tolist())

    def test_facility_bundle_mode_tie_and_no_inflated_counts(self):
        base = raw_row(**{"위도": "37.5", "경도": "127.1", "학교_시설명": "학교 A", "학교_거리_m": "0"})
        other = dict(base, **{"학교_시설명": "학교 B", "학교_거리_m": "500"})
        for rows in [[base, other], [other, base, base]]:
            data = prepare_data(pd.DataFrame(rows))
            school = data.facilities.loc[data.facilities.kind == "학교"].iloc[0]
            self.assertEqual(school["name"], "학교 A")
            self.assertEqual(school.distance, 0)
            self.assertTrue(pd.isna(school["count"]))
            self.assertEqual(len(data.facilities), 4)

    def test_coordinate_missing_differs_from_unmapped(self):
        for coords, expected in [({}, "no_coordinates"), ({"위도": "37.5", "경도": "127.1"}, "unmapped")]:
            data = prepare_data(pd.DataFrame([raw_row(**coords)]))
            self.assertEqual(data.facilities.status.iloc[0], expected)

    def test_facility_over_1km_names_the_kind_instead_of_hiding_it(self):
        # 이름·좌표는 있지만 1km를 넘는 경우: 거리는 숨기되 "왜" 안 보이는지는 사실대로 말한다.
        row = raw_row(**{"위도": "37.5", "경도": "127.1", "공원_시설명": "테스트근린공원",
                         "공원_위도": "37.51", "공원_경도": "127.11", "공원_거리_m": "1318"})
        data = prepare_data(pd.DataFrame([row]))
        facility = data.facilities.loc[data.facilities.kind == "공원"].iloc[0]
        self.assertEqual(facility.status, "invalid")
        s = build_snapshot(data, data.complexes.complex_id.iloc[0], AREA_LABELS[2])
        markup = comparison_details(s)
        self.assertIn("테스트근린공원", markup)  # 이름은 계속 보여준다
        self.assertIn("1km 밖", markup)  # 옛 "거리 확인 필요"는 더 이상 안 쓴다
        self.assertNotIn("확인 필요", markup)

    def test_josa_i_ga_picks_batchim_correctly(self):
        self.assertEqual(josa_i_ga("지하철"), "이")  # 받침 ㄹ
        self.assertEqual(josa_i_ga("학교"), "가")    # 받침 없음
        self.assertEqual(josa_i_ga("공원"), "이")    # 받침 ㄴ
        self.assertEqual(josa_i_ga("장보기"), "가")  # 받침 없음
        self.assertEqual(josa_i_ga(""), "가")        # 빈 문자열도 안전하게 처리

    def test_complex_details_preserve_zero_missing_and_separate_bundles(self):
        first = raw_row(**{"건축년도": "2008", "단지_세대수": "5,563", "단지_지상주차대수": "0",
                           "단지_지하주차대수": "6,000", "단지_복도유형": " 계단식 "})
        other = raw_row(**{"단지명": "다른 단지", "도로명주소": "다른 주소", "단지_복도유형": "-"})
        data = prepare_data(pd.DataFrame([first, first, other]))
        original = data.complexes.loc[data.complexes.name == "테스트"].iloc[0]
        self.assertEqual(original.built_year, 2008)
        self.assertEqual(original.households, 5563)
        self.assertEqual(original.parking_ground, 0)
        self.assertEqual(original.parking_underground, 6000)
        self.assertEqual(original.corridor_type, "계단식")
        missing = data.complexes.loc[data.complexes.name == "다른 단지"].iloc[0]
        self.assertIsNone(missing.corridor_type)
        self.assertTrue(pd.isna(missing.parking_ground))

    def test_geodesic_ring_and_no_coordinates_map(self):
        ring = radius_polygon(37.5, 127.1)
        self.assertEqual(len(ring), 73)
        self.assertAlmostEqual(ring[0][1] - 37.5, 1000 / 6371008.8 * 180 / np.pi)
        s = build_snapshot(fixture([100, 120, 80, 130]), "unit", AREA_LABELS[2])
        self.assertIsNone(facility_map(s))


class SourceAcceptanceTests(unittest.TestCase):
    """Real-source checks run on a committed fixture so they stay fast and independent of the shipped CSV."""

    @classmethod
    def setUpClass(cls):
        cls.before = sha256(FIXTURE_CSV.read_bytes()).hexdigest()
        cls.data = load_csv(FIXTURE_CSV)

    @unittest.skipUnless(DEFAULT_CSV.exists(), "실데이터 파일은 포트폴리오 저장소에 포함하지 않음")
    def test_default_csv_is_reachable(self):
        self.assertTrue(DEFAULT_CSV.exists(), f"기본 CSV 경로가 잘못되었습니다: {DEFAULT_CSV}")

    def test_fixture_pipeline_and_input_unchanged(self):
        data = self.data
        self.assertEqual(data.diagnostics["원본 행"], 17)
        self.assertEqual(data.diagnostics["유효 거래"], 17)
        self.assertEqual(data.diagnostics["KAPT 코드 결측"], 2)
        self.assertEqual(data.diagnostics["고유 단지"], 3)
        self.assertEqual(str(data.start.date()), "2020-01-01")
        self.assertEqual(str(data.end.date()), "2021-01-15")
        self.assertEqual(data.reference, pd.Period("2021Q1"))
        self.assertTrue(data.partial)
        self.assertEqual(sorted(data.complexes.district.unique()), ["강남구", "송파구"])
        self.assertEqual(sha256(FIXTURE_CSV.read_bytes()).hexdigest(), self.before)

    def test_fixture_metrics_are_hand_verifiable(self):
        # Historical complete-quarter math remains verifiable independently of the latest reference policy.
        s = build_snapshot(replace(self.data, reference=pd.Period("2020Q4"), partial=False), "kapt:A13822003", AREA_LABELS[2])
        # Quarterly medians are 100, 120, 80, 130 만원/㎡ over 2020Q1~2020Q4.
        self.assertEqual(s.metrics["price"].value, 130)
        self.assertEqual(s.metrics["quarter_count"].value, 3)
        self.assertEqual(s.metrics["year_count"].value, 12)
        self.assertAlmostEqual(s.metrics["mdd"].value, -100 / 3)
        self.assertEqual(s.metrics["peak"].value, 120)
        self.assertEqual(s.metrics["peak"].period, "2020Q2")
        self.assertEqual(s.metrics["trough"].value, 80)
        self.assertEqual(s.metrics["trough"].period, "2020Q3")
        self.assertAlmostEqual(s.metrics["gap"].value, 100 / 12)
        self.assertEqual(s.metrics["recovery"].value, 125)
        self.assertEqual(s.metrics["recovery_quarters"].value, 1)
        # 등급은 같은 면적군 안의 순위라, 픽스처처럼 비교 대상이 자기 하나뿐이면 1/1위가 된다.
        self.assertEqual((s.score["rank"], s.score["peers"]), (1, 1))
        transit = s.facilities.loc[s.facilities.kind == "지하철"].iloc[0]
        self.assertEqual(transit["name"], "잠실새내")
        self.assertEqual(transit.distance, 650.3)
        self.assertEqual(s.facilities.loc[s.facilities.kind == "대규모점포"].iloc[0].status, "unmapped")

    def test_map_pixel_units_serialize_as_constants(self):
        s = build_snapshot(self.data, "kapt:A13822003", AREA_LABELS[2])
        deck = json.loads(facility_map(s).to_json())
        pins = deck["layers"][1]
        self.assertEqual(deck["layers"][0]["lineWidthUnits"], "pixels")
        self.assertEqual(pins["@@type"], "IconLayer")
        self.assertEqual(pins["sizeUnits"], "pixels")
        self.assertEqual(len(pins["data"]), 4)  # Apartment + three mapped nearest facilities.

    def test_demo_matches_plan_and_is_deterministic(self):
        data = demo_dataset()
        pd.testing.assert_frame_equal(data.quarterly, demo_dataset().quarterly)
        s = build_snapshot(data, "demo:리센츠", AREA_LABELS[2])
        self.assertEqual(s.metrics["price"].value, 4118)
        self.assertEqual(s.metrics["peak"].value, 3540)
        self.assertEqual(s.metrics["trough"].value, 2389)
        self.assertEqual(round(s.metrics["mdd"].value, 1), -32.5)
        self.assertEqual(round(s.metrics["gap"].value, 1), 16.3)
        self.assertEqual(round(s.metrics["recovery"].value, 1), 150.2)
        self.assertEqual(s.metrics["quarter_count"].value, 89)
        self.assertEqual(s.metrics["year_count"].value, 205)
        self.assertEqual((s.score["grade"], s.score["total"]), ("B+", 64))
        self.assertEqual((s.score["defense"], s.score["recovery"]), (6, 58))
        for cid in data.complexes.complex_id:
            snap = build_snapshot(data, cid, AREA_LABELS[2])
            self.assertEqual(snap.metrics["year_count"].value, snap.series["count"].tail(4).sum())
            self.assertEqual(snap.metrics["price"].value, snap.series.price.iloc[-1])
        for name, fall, gap in [("헬리오시티", -26.0, 17.6), ("파크리오", -25.2, 24.1)]:
            snap = build_snapshot(data, f"demo:{name}", AREA_LABELS[2])
            self.assertEqual(round(snap.metrics["mdd"].value, 1), fall)
            self.assertEqual(round(snap.metrics["gap"].value, 1), gap)


class AppTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"APARTMENT_DATA_CSV": str(FIXTURE_CSV)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self):
        return AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=60).run()

    def test_initial_real_mode_and_atomic_apply(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="view_mode").value, "실데이터")
        self.assertEqual(app.selectbox(key="real_complex").value, "kapt:A13822003")
        app.selectbox(key="real_complex").select("kapt:A13824006").run()
        self.assertEqual(app.session_state["domain_real"]["applied"]["complex_id"], "kapt:A13822003")
        app.button[0].click().run()
        self.assertEqual(app.session_state["domain_real"]["applied"]["complex_id"], "kapt:A13824006")
        self.assertFalse(app.exception)

    def test_mode_state_comparison_and_no_leak(self):
        app = self.app()
        app.radio(key="view_mode").set_value("시연 화면").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.multiselect(key="demo_extras").value, ["demo:헬리오시티", "demo:파크리오"])
        app.multiselect(key="demo_extras").set_value(["demo:문정시영"]).run()
        app.selectbox(key="demo_complex").select("demo:파크리오").run()
        app.button[0].click().run()
        app.radio(key="view_mode").set_value("실데이터").run()
        self.assertEqual(app.session_state["domain_real"]["applied"]["complex_id"], "kapt:A13822003")
        all_html = "\n".join(element.value for element in app.markdown)
        self.assertNotIn("4,118만원", all_html)
        # The method popover legitimately contains the 80-point recovery maximum.
        # Check the ring's actual total rather than any occurrence of that text.
        self.assertNotIn("<b>80점</b>", all_html)
        self.assertIn("<b>50점</b>", all_html)
        app.radio(key="view_mode").set_value("시연 화면").run()
        self.assertEqual(app.selectbox(key="demo_complex").value, "demo:파크리오")
        self.assertEqual(app.multiselect(key="demo_extras").value, ["demo:문정시영"])
        self.assertFalse(app.exception)

    def test_district_change_clears_invalid_comparisons(self):
        app = self.app()
        app.multiselect(key="real_extras").set_value(["kapt:A13824006"]).run()
        app.selectbox(key="real_district").select("강남구").run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["domain_real"]["applied"]["district"], "강남구")
        self.assertEqual(app.multiselect(key="real_extras").value, [])

    def test_file_replacement_refreshes_cache_and_selection(self):
        qa = APP_DIR / ".qa"
        qa.mkdir(exist_ok=True)
        path = qa / f"test-input-{uuid4().hex}.csv"
        try:
            pd.DataFrame([raw_row()]).to_csv(path, index=False, encoding="utf-8-sig")
            with patch.dict(os.environ, {"APARTMENT_DATA_CSV": str(path)}):
                app = self.app()
                self.assertFalse(app.exception)
                first_id = app.selectbox(key="real_complex").value
                pd.DataFrame([raw_row(**{"단지명": "교체된 다른 단지", "계약년월": "2021-07"})]).to_csv(path, index=False, encoding="utf-8-sig")
                app.run()
                self.assertFalse(app.exception)
                self.assertNotEqual(app.selectbox(key="real_complex").value, first_id)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_file_keeps_demo_accessible(self):
        with patch.dict(os.environ, {"APARTMENT_DATA_CSV": str(APP_DIR / "missing.csv")}):
            app = self.app()
            self.assertEqual(len(app.error), 1)
            self.assertFalse(app.exception)
            app.radio(key="view_mode").set_value("시연 화면").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 0)


if __name__ == "__main__":
    unittest.main()
