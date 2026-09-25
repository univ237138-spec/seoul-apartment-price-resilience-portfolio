from dataclasses import dataclass, field

import pandas as pd


AREA_BINS = [0, 40, 60, 85, 102, 135, float("inf")]
AREA_LABELS = [
    "40㎡ 미만", "40㎡ 이상 60㎡ 미만", "60㎡ 이상 85㎡ 미만",
    "85㎡ 이상 102㎡ 미만", "102㎡ 이상 135㎡ 미만", "135㎡ 이상",
]
FACILITY_TYPES = ["지하철", "학교", "공원", "대규모점포"]


@dataclass(frozen=True)
class Metric:
    value: float | int | None
    unit: str
    period: str
    source: str
    status: str = "ok"
    reason: str = ""


@dataclass
class Dataset:
    source: str
    quarterly: pd.DataFrame
    complexes: pd.DataFrame
    facilities: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    reference: pd.Period
    partial: bool
    diagnostics: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    representative_areas: dict = field(default_factory=dict)
    last_trades: dict = field(default_factory=dict)  # (단지, 면적군) -> 마지막 실거래 한 건


@dataclass
class Snapshot:
    complex_id: str
    area_group: str
    source: str
    reference: pd.Period  # 자료 전체의 최신 분기
    basis: pd.Period  # 가격·지표를 실제로 계산한 분기. 최신 분기가 얕으면 뒤로 물러난다.
    partial: bool
    metrics: dict[str, Metric]
    series: pd.DataFrame
    info: dict
    facilities: pd.DataFrame
    score: dict | None = None


def duration(quarters) -> str:
    """Quarters as 년·개월, which reads faster than a quarter count for a buyer."""
    if quarters is None:
        return "자료 부족"
    years, months = divmod(int(quarters) * 3, 12)
    parts = [f"{years}년" if years else "", f"{months}개월" if months else ""]
    return " ".join(part for part in parts if part) or "0개월"


def quarter_label(value) -> str:
    if value is None or value == "":
        return "시점 정보 없음"
    q = pd.Period(value, freq="Q")
    return f"{q.year}년 {q.quarter}분기"
