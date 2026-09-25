"""Pure calculations shared by the real and demonstration screens."""
import numpy as np
import pandas as pd

from .models import Dataset, Metric, Snapshot


MIN_QUARTER_TRADES = 3
MIN_COMPLETE_QUARTERS = 4


def drawdown(prices: pd.Series):
    """Deepest peak-to-trough decline and the peak that preceded it.

    Shared with the grading pass so both read one definition: the first chronological
    trough wins equal drawdowns, and a later new high never replaces the earlier peak.
    Returns depth alone when no decline was observed, since there is no pair to report.
    """
    running = (prices / prices.cummax() - 1) * 100
    trough_q = running.idxmin()
    depth = float(running.loc[trough_q])
    if np.isclose(depth, 0):
        return depth, None, None, None, None
    peak_q = prices.loc[:trough_q].idxmax()
    return depth, peak_q, float(prices.loc[peak_q]), trough_q, float(prices.loc[trough_q])


def build_snapshot(data: Dataset, complex_id: str, area_group: str) -> Snapshot:
    q = data.quarterly
    observed = q.loc[(q.complex_id == complex_id) & (q.area_group == area_group)
                     & (q.quarter <= data.reference)].sort_values("quarter").copy()
    source = data.source

    complete_qualified = observed.complete & observed["count"].ge(MIN_QUARTER_TRADES)
    eligible = observed.complete | (data.partial & observed.quarter.eq(data.reference))
    observed["qualified"] = eligible & observed["count"].ge(MIN_QUARTER_TRADES)
    qualified = observed.loc[observed["qualified"]].copy()
    # 최신 분기의 거래가 얕으면 가격 기준을 마지막으로 3건 이상 거래된 분기까지 물린다. 없는 값을
    # 지어내는 대신 관측된 분기를 그대로 쓰고, 그게 언제인지는 basis로 넘겨 화면이 밝히게 한다.
    basis_row = qualified.iloc[-1] if len(qualified) else None
    basis = basis_row.quarter if basis_row is not None else data.reference
    period = str(basis)

    def metric(value=None, unit="%", status="ok", reason="", at=period):
        return Metric(value, unit, str(at), source, status, reason)

    # 잠정 표시는 기준이 실제로 미완결 분기일 때만 붙인다. 뒤로 물러난 기준은 완결 분기다.
    display_status = "partial" if data.partial and basis == data.reference else "ok"
    absent = "거래 3건 이상인 분기가 없어 기준 가격을 정하지 않았습니다."
    m = {
        "price": metric(float(basis_row.price), "만원/㎡", display_status)
        if basis_row is not None else metric(unit="만원/㎡", status="missing", reason=absent),
        "quarter_count": metric(int(basis_row["count"]) if basis_row is not None else 0, "건", display_status),
    }
    annual_start = (data.reference - 3).start_time
    # 최근 1년 거래량은 시장의 최신 활발함을 말하므로 기준 분기가 아니라 자료의 최신 분기에 건다.
    annual_covered = data.start.normalize() <= annual_start
    m["year_count"] = (
        metric(int(observed.loc[observed.quarter >= data.reference - 3, "count"].sum()), "건",
               "partial" if data.partial else "ok", at=str(data.reference))
        if annual_covered
        else metric(unit="건", status="insufficient", reason="최근 4개 분기를 포함하는 관측 범위가 없습니다.",
                    at=str(data.reference))
    )
    reason = "분기당 3건 이상인 완결 분기가 4개 이상 필요합니다. 최신 분기가 얕으면 마지막으로 3건 이상 거래된 분기를 기준으로 씁니다."
    units = {"mdd": "%", "peak": "만원/㎡", "trough": "만원/㎡", "gap": "%",
             "recovery": "%", "recovery_quarters": "분기"}
    for key, unit in units.items():
        m[key] = metric(unit=unit, status="insufficient", reason=reason)
    if int(complete_qualified.sum()) >= MIN_COMPLETE_QUARTERS and basis_row is not None:
        prices = qualified.set_index("quarter")["price"]
        mdd, peak_q, peak, trough_q, trough = drawdown(prices)
        m["mdd"] = metric(mdd, status=display_status)
        if peak_q is None:
            for key in ["peak", "trough", "gap", "recovery", "recovery_quarters"]:
                m[key] = metric(unit=units[key], status="no_decline", reason="관측된 하락 구간이 없습니다.")
        else:
            price = m["price"].value
            m["peak"] = metric(peak, "만원/㎡", at=peak_q)
            m["trough"] = metric(trough, "만원/㎡", at=trough_q)
            m["gap"] = metric((price / peak - 1) * 100, status=display_status)
            m["recovery"] = metric((price - trough) / (peak - trough) * 100, status=display_status)
            recovered = prices.loc[(prices.index > trough_q) & prices.ge(peak)]
            m["recovery_quarters"] = (
                metric(int(recovered.index[0].ordinal - trough_q.ordinal), "분기", at=recovered.index[0],
                       status="partial" if data.partial and recovered.index[0] == data.reference else "ok")
                if len(recovered) else metric(unit="분기", status="unrecovered", reason="기준 고점을 아직 회복하지 못했습니다.")
            )
    # Reindex missing quarters as NaN so the real chart never bridges a missing observation.
    first = data.start.to_period("Q")
    series = observed.set_index("quarter").reindex(pd.period_range(first, data.reference, freq="Q"))
    series.index.name = "quarter"
    info = data.complexes.set_index("complex_id").loc[complex_id].to_dict()
    info["median_area"] = data.representative_areas.get((complex_id, area_group))
    return Snapshot(complex_id, area_group, source, data.reference, basis, data.partial, m,
                    series.reset_index(), info,
                    data.facilities.loc[data.facilities.complex_id == complex_id].copy(),
                    data.scores.get((complex_id, area_group)))


def narrative(s: Snapshot) -> dict:
    m = s.metrics
    mdd, gap, recovery = (m[k].value for k in ["mdd", "gap", "recovery"])
    if mdd is None:
        return {
            "headline": "아직은 가격 회복력을\n판단할 기록이 부족해요.",
            "summary": "확인된 가격과 생활시설부터 살펴보세요. 더 긴 거래 이력이 쌓이면 회복 흐름을 비교할 수 있어요.",
            "check_title": "거래 기간을 먼저 확인해요",
            "check": "현재 자료로는 장기 하락과 회복을 판단하기 어려워요. 표시된 가격은 제공된 거래 표본의 값이에요.",
            "positive": "확인된 거래의 ㎡당 중위가격과 주변 시설의 직선거리를 살펴볼 수 있어요.",
            "watch": "분기별 거래 기록이 더 필요해요. 자료 부족은 하락이 없었다는 뜻이 아니에요.",
            "badge": "장기 자료 부족",
        }
    if gap is None:
        headline, badge = "관측된 기간에는\n가격 하락 구간이 없었어요.", "하락 구간 없음"
    elif gap >= 0:
        headline = "큰 하락을 겪었지만,\n지금은 이전 고점을 넘었어요." if abs(mdd) >= 30 else "하락 이후 회복하며\n이전 고점을 되찾았어요."
        badge = "전고점 돌파" if gap > 0 else "전고점 회복"
    elif recovery >= 80:
        headline, badge = "하락분을 많이 회복해\n이전 고점에 가까워졌어요.", "전고점 근접"
    elif recovery >= 50:
        headline, badge = "저점에서 올라왔지만,\n아직 회복이 진행 중이에요.", "회복 진행"
    else:
        headline, badge = "이전 고점까지는\n회복할 거리가 남아 있어요.", "회복 지연"
    return {
        "headline": headline, "badge": badge,
        "summary": f'{s.info["name"]}의 실거래 기록을 구매자의 눈높이로 요약했어요.' if s.source == "real" else f'{s.info["name"]}의 예시 가격 흐름을 구매자의 눈높이로 살펴보세요.',
        "check_title": "“안전하다”는 뜻은 아니에요",
        "check": "회복 정도와 과거 하락 폭을 함께 살펴보세요. 내가 감당할 수 있는 가격 변동인지 확인하는 데 도움이 돼요.",
        "positive": "하락 후 이전 고점을 넘어선 가격이 다시 관측되었어요." if gap is not None and gap >= 0 else "분기별 가격과 거래량을 함께 보며 회복 과정을 비교할 수 있어요.",
        "watch": "과거에 30% 이상 떨어진 적이 있어 변동성이 작다고 보기는 어려워요." if abs(mdd) >= 30 else "과거 회복이 앞으로도 반복된다고 볼 수는 없어요. 거래량과 가격 흐름을 함께 확인하세요.",
    }
