"""거래가 얇은 조합에 나가는 대체 화면과, 어느 화면이 나갈지 고르는 분기를 고정한다."""
from dataclasses import replace
import unittest

import pandas as pd

from test_dashboard import fixture, raw_row
from housing.data import prepare_data
from housing.metrics import build_snapshot
from housing.models import AREA_LABELS
from housing.sparse_overview import sparse_values
from housing.sparse_overview_ui import won

AREA = AREA_LABELS[2]


def thin_dataset():
    """완결 분기 4개를 못 채우는 조합. 거래는 있으나 낙폭·회복률은 못 낸다."""
    rows = []
    for month, count, price in [("2025-04", 5, 100), ("2025-07", 3, 110), ("2025-10", 1, 130)]:
        for _ in range(count):
            # 85.0은 다음 면적군(85㎡ 이상)으로 넘어가므로 구간 안쪽 값을 쓴다.
            rows.append(raw_row(**{"계약년월": month, "거래금액(만원)": str(round(price * 84.9)),
                                   "전용면적(㎡)": "84.9"}))
    return prepare_data(pd.DataFrame(rows))


class BranchTests(unittest.TestCase):
    def test_full_metrics_and_thin_data_take_different_screens(self):
        # 화면 선택은 mdd 하나로 갈린다. 이 신호가 바뀌면 분기도 같이 흔들린다.
        rich = build_snapshot(fixture([100, 120, 80, 130]), "unit", AREA)
        self.assertIsNotNone(rich.metrics["mdd"].value)  # -> 기존 화면
        data = thin_dataset()
        thin = build_snapshot(data, data.complexes.complex_id.iloc[0], AREA)
        self.assertIsNone(thin.metrics["mdd"].value)  # -> 대체 화면


class SparseValueTests(unittest.TestCase):
    def setUp(self):
        self.data = thin_dataset()
        self.s = build_snapshot(self.data, self.data.complexes.complex_id.iloc[0], AREA)
        self.view = sparse_values(self.data, self.s)

    def test_counts_quarters_since_the_first_trade_not_the_whole_file(self):
        # 2025Q2에 처음 거래된 단지에 자료 전체 분기를 분모로 들이대면 실제보다 부실해 보인다.
        self.assertEqual(self.view["span"], 3)  # 2025Q2~2025Q4
        self.assertEqual(self.view["qualified"], 2)  # 5건·3건 분기만 자격
        self.assertEqual(self.view["first_period"], "2025년 4~6월")

    def test_latest_trade_is_a_real_contract_not_a_quarter_median(self):
        trade = self.view["latest_trade"]
        self.assertEqual(trade["amount"], round(130 * 84.9))  # 마지막 분기의 실제 체결가
        self.assertEqual(trade["date"], "2025년 10월 1일")
        self.assertAlmostEqual(trade["unit_price"], 130)

    def test_change_reads_first_to_last_observed_quarter(self):
        change = self.view["change"]
        self.assertAlmostEqual(change["pct"], 30.0)
        self.assertEqual((change["from_price"], change["to_price"]), (100, 130))

    def test_single_quarter_reports_no_change(self):
        # 관측 분기가 하나뿐이면 변화를 말할 수 없다. 0%라고 지어내지 않는다.
        rows = [raw_row(**{"거래금액(만원)": "8490", "전용면적(㎡)": "84.9"}) for _ in range(2)]
        data = prepare_data(pd.DataFrame(rows))
        s = build_snapshot(data, data.complexes.complex_id.iloc[0], AREA)
        self.assertIsNone(sparse_values(data, s)["change"])


class AmountTests(unittest.TestCase):
    def test_won_splits_billions_from_the_remainder(self):
        self.assertEqual(won(161000), "16억 1,000<small>만원</small>")
        self.assertEqual(won(120000), "12<small>억원</small>")
        self.assertEqual(won(9800), "9,800<small>만원</small>")


if __name__ == "__main__":
    unittest.main()
