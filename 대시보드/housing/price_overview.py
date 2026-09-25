"""Source-derived values for the UI v2 price cards."""
import math

import pandas as pd

from .models import AREA_BINS, AREA_LABELS, Dataset, Snapshot, duration


def month_label(period, data):
    q = pd.Period(period, freq="Q")
    start = max(q.start_time, data.start)
    end = min(q.end_time, data.end)
    months = f"{start.month}월" if start.month == end.month else f"{start.month}~{end.month}월"
    return f"{q.year}년 {months}"


def overview_values(data: Dataset, s: Snapshot):
    m = s.metrics
    upper = AREA_BINS[AREA_LABELS.index(s.area_group) + 1]
    area = upper if math.isfinite(upper) else s.info.get("median_area")
    if area is not None and not math.isfinite(area):
        area = None
    peak, trough = m["peak"], m["trough"]
    has_pair = peak.value is not None and trough.value is not None
    decline = pd.Period(trough.period, freq="Q").ordinal - pd.Period(peak.period, freq="Q").ordinal if has_pair else None
    elapsed = s.basis.ordinal - pd.Period(peak.period, freq="Q").ordinal if has_pair else None
    recovery = m["recovery_quarters"]
    recovery_text = duration(recovery.value) + " 만에 첫 회복" if recovery.value is not None else (
        "아직 미회복" if recovery.status == "unrecovered" else "하락 구간 없음" if recovery.status == "no_decline" else "자료 부족")
    if recovery.status == "partial":
        recovery_text += " · 잠정"
    qualified = int(s.series["qualified"].eq(True).sum())
    stale = s.basis < s.reference
    if m["price"].value is None:
        status = "거래 3건 이상인 분기 없음"
    elif m["mdd"].value is None:
        status = "분기 기록 부족"
    elif not has_pair:
        status = "하락 구간 없음"
    else:
        status = "고점·저점 계산 가능"
    missing_period = "하락 구간 없음" if peak.status == "no_decline" else "자료 부족"
    method = "완결 분기 4개 이상 확보 후, 거래 3건 이상인 분기를 사용합니다. 회복 기간은 저점 이후 최초 고점 회복까지의 분기 수로 계산합니다."
    if stale:
        method += f" 최신 {month_label(s.reference, data)}은 거래 3건에 못 미쳐, 마지막으로 3건 이상 거래된 {month_label(s.basis, data)}을 기준으로 계산했습니다."
    elif s.partial:
        method += " 현재는 부분 분기 잠정값으로 추가 거래에 따라 달라질 수 있습니다."
    if m["mdd"].value is None:
        method += " " + m["mdd"].reason
    return {
        "area": area, "peak": peak.value, "trough": trough.value, "price": m["price"].value,
        "mdd": m["mdd"].value, "gap": m["gap"].value,
        "peak_period": month_label(peak.period, data) if has_pair else missing_period,
        "trough_period": month_label(trough.period, data) if has_pair else missing_period,
        "current_period": month_label(s.basis, data),
        "current_label": "마지막 확인" if stale else "현재·잠정" if s.partial else "현재",
        "stale": stale, "latest_period": month_label(s.reference, data),
        "decline_duration": duration(decline), "since_peak": duration(elapsed), "recovery_duration": recovery_text,
        "count": m["quarter_count"].value, "qualified": qualified, "total_quarters": len(s.series), "status": status,
        "subtitle": f'{s.info["name"]} · {s.info["dong"]} · {s.area_group}',
        "years": f"{data.start.year} → {s.reference.year}", "source_label": "실거래" if s.source == "real" else "예시 거래",
        "method_note": method,
    }
