"""Source-derived values for the screen shown when a combination cannot carry the full metrics.

낙폭·회복률은 완결 분기 4개를 요구하므로, 이력이 짧은 단지(대개 신축)에서는 계산되지 않는다.
그때 화면을 "자료 부족" 한 줄로 비우는 대신, 실제로 기록된 것만 골라 보여준다. 이 모듈은
없는 값을 채우지 않고, quarterly와 마지막 실거래에서 읽을 수 있는 것만 옮긴다.
"""
import pandas as pd

from .models import Dataset, Snapshot
from .price_overview import month_label


def sparse_values(data: Dataset, s: Snapshot) -> dict:
    series = s.series.dropna(subset=["price"])
    first, last = series.iloc[0], series.iloc[-1]
    qualified = int(s.series["qualified"].eq(True).sum())
    # 분모는 자료 전체가 아니라 이 조합의 거래가 처음 확인된 뒤로 흐른 분기다. 2024년 준공 단지에
    # 39분기를 분모로 들이대면 실제보다 부실해 보인다.
    span = (s.reference - first.quarter).n + 1

    trade = data.last_trades.get((s.complex_id, s.area_group))
    latest_trade = None
    if trade is not None:
        day = trade["date"]
        latest_trade = {
            "amount": float(trade["amount"]), "area": float(trade["area"]),
            "unit_price": float(trade["unit_price"]),
            "date": f"{day.year}년 {day.month}월 {day.day}일",  # 0을 뗀 표기는 플랫폼별 지시자 대신 직접 만든다
        }

    change = None
    if len(series) >= 2 and float(first.price) > 0:
        change = {
            "pct": (float(last.price) / float(first.price) - 1) * 100,
            "from_price": float(first.price), "to_price": float(last.price),
            "from_period": month_label(first.quarter, data), "to_period": month_label(last.quarter, data),
        }

    return {
        "subtitle": f'{s.info["name"]} · {s.info["dong"]} · {s.area_group}',
        "years": f"{first.quarter.year} → {s.reference.year}",
        "first_period": month_label(first.quarter, data),
        "qualified": qualified, "span": span,
        "latest_trade": latest_trade, "change": change,
        "source_label": "실거래" if s.source == "real" else "예시 거래",
    }
