"""District-and-area distribution using the dashboard's existing metric rules."""
import numpy as np
import pandas as pd

from .metrics import MIN_QUARTER_TRADES
from .models import Dataset, Snapshot
from .scoring import axis_inputs


def distribution_values(data: Dataset, selected: Snapshot) -> dict:
    district = selected.info["district"]
    info = data.complexes.loc[data.complexes.district.eq(district)]
    q = data.quarterly.loc[
        data.quarterly.complex_id.isin(info.complex_id)
        & data.quarterly.area_group.eq(selected.area_group)
        & data.quarterly.quarter.le(data.reference)
    ]
    candidates = q.complex_id.nunique()
    axes = axis_inputs(q, data.reference, data.partial)
    finite = np.isfinite(axes["mdd"].astype(float)) & np.isfinite(axes["gap"].astype(float))
    points = axes.loc[finite, ["complex_id", "mdd", "gap"]].merge(
        info[["complex_id", "name", "dong", "address"]], on="complex_id", validate="one_to_one")
    qualified = q.loc[q["count"].ge(MIN_QUARTER_TRADES)
                      & (q.complete | (data.partial & q.quarter.eq(data.reference)))]
    latest = qualified.sort_values("quarter").groupby("complex_id", observed=True).tail(1)
    points = points.merge(latest[["complex_id", "quarter", "count"]], on="complex_id", validate="one_to_one")
    points = points.rename(columns={"quarter": "basis"}).sort_values(["name", "complex_id"]).reset_index(drop=True)
    points["partial"] = data.partial & points.basis.eq(data.reference)
    peers = points.loc[points.complex_id.ne(selected.complex_id)]
    own = points.loc[points.complex_id.eq(selected.complex_id)]
    return {
        "district": district, "area_group": selected.area_group, "source": data.source,
        "selected_id": selected.complex_id, "selected_name": selected.info["name"],
        "selected": own.iloc[0].to_dict() if len(own) else None,
        "selected_status": selected.metrics["gap"].status,
        "points": points, "candidates": candidates,
        "insufficient": candidates - len(axes), "no_decline": int(axes.gap.isna().sum()),
        "other_count": len(peers), "stale_count": int(points.basis.lt(data.reference).sum()),
        "median_mdd": float(peers.mdd.median()) if len(peers) else None,
        "median_gap": float(peers.gap.median()) if len(peers) else None,
    }


def distribution_narrative(view: dict) -> tuple[str, str]:
    own = view["selected"]
    if own is None:
        reason = ("관측된 하락 구간이 없어 고점 대비 위치를 계산할 수 없어요."
                  if view["selected_status"] == "no_decline" else "장기 지표를 산출할 거래 기록이 부족해요.")
        return "선택 단지는 분포에 표시하지 않았어요", reason + " 조건을 충족한 같은 구·면적 단지만 표시해요."
    if not view["other_count"]:
        return "아직 비교할 다른 단지가 없어요", "선택 단지의 좌표만 표시해요. 비교 가능한 거래 기록이 쌓이면 상대 위치도 안내할게요."

    dx, dy = own["mdd"] - view["median_mdd"], own["gap"] - view["median_gap"]
    defense = "낙폭이 비슷하고" if np.isclose(dx, 0) else "덜 떨어졌고" if dx > 0 else "더 크게 떨어졌고"
    recovery = "현재 위치도 비슷해요" if np.isclose(dy, 0) else "현재 위치는 더 높아요" if dy > 0 else "현재 위치는 더 낮아요"
    headline = f'{view["selected_name"]}는 비교 단지 중앙값보다 {defense}, {recovery}.'
    body = (f'선택 단지를 제외한 {view["other_count"]:,}곳의 중앙값은 낙폭 {view["median_mdd"]:.1f}%, '
            f'고점 대비 {view["median_gap"]:+.1f}%예요. 각 단지의 과거 고점을 기준으로 비교해요.')
    if view["other_count"] < 5:
        body += " 비교 단지가 적어 상대 위치 해석에 유의해 주세요."
    return headline, body
