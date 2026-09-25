"""Peer-relative grade for the summary ring.

An absolute cutoff — "MDD −40% is zero points" — has no market-defined anchor, so each
axis is ranked against the complexes of the same area group instead. A score then states
something that is always true of the data: where this combination sits among comparable
candidates. Only the scale comes from the data; the 20/80 split between decline and
recovery is a stated editorial choice, recorded here so it can be argued with.

The recovery axis reads 고점 대비 위치 (gap), not 하락분 회복률. The 회복률 denominator
(고점 − 저점) collapses toward zero on a shallow decline and the ratio then explodes —
the EDA over the full CSV found values up to 2,718,940%, with 86.9% of readings above
1,000% coming from combinations whose |MDD| is under 5%. gap has no such instability.
"""
import pandas as pd

from .metrics import MIN_COMPLETE_QUARTERS, MIN_QUARTER_TRADES, drawdown

DEFENSE_WEIGHT = 20
RECOVERY_WEIGHT = 80
# A grade names a rank, not an absolute standard: A+ is the top tenth of scored combinations.
GRADE_BANDS = [(90, "A+"), (75, "A"), (55, "B+"), (35, "B"), (15, "C+"), (0, "C")]


def rank_percentile(values: pd.Series, value: float) -> float:
    """Share of peers this value beats, splitting ties so equal values share one band."""
    if values.empty:
        return 100.0
    beaten = (values < value).sum() + 0.5 * (values == value).sum()
    return float(beaten / len(values) * 100)


def grade_of(percentile: float) -> str:
    for floor, label in GRADE_BANDS:
        if percentile >= floor:
            return label
    return GRADE_BANDS[-1][1]


def axis_inputs(quarterly: pd.DataFrame, reference, partial: bool) -> pd.DataFrame:
    """MDD and 고점 대비 위치 for every combination the dashboard would show metrics for.

    Mirrors the gate in build_snapshot so a complex is never graded on a screen that
    reports 자료 부족: four complete quarters carrying at least three trades each. The
    price both axes read is the last qualifying quarter's — the same basis the card
    shows — so a grade never describes a quarter the screen doesn't name.
    """
    q = quarterly.loc[quarterly.quarter <= reference]
    keys = ["complex_id", "area_group"]
    enough = q["count"].ge(MIN_QUARTER_TRADES)
    complete_qualified = q.loc[q.complete & enough].groupby(keys, observed=True).size()
    eligible = q.complete | (partial & q.quarter.eq(reference))

    rows = []
    for key, group in q.loc[eligible & enough].groupby(keys, observed=True, sort=False):
        if complete_qualified.get(key, 0) < MIN_COMPLETE_QUARTERS:
            continue
        prices = group.sort_values("quarter").set_index("quarter")["price"]
        mdd, _, peak, _, trough = drawdown(prices)
        complex_id, area_group = key
        rows.append({"complex_id": complex_id, "area_group": area_group, "mdd": mdd,
                     "gap": None if peak is None else (float(prices.iloc[-1]) / peak - 1) * 100,
                     "recovery_ratio": None if peak is None else
                     (float(prices.iloc[-1]) - trough) / (peak - trough) * 100})
    return pd.DataFrame(rows, columns=["complex_id", "area_group", "mdd", "gap", "recovery_ratio"])


def build_scores(quarterly: pd.DataFrame, reference, partial: bool) -> dict:
    """Grade every scorable combination against its own area group.

    Ranking runs per area group because that is the comparison a buyer is actually making:
    the EDA showed 지표 표시 가능률 varies from 4.7% to 23.0% across area groups, so pooling
    them would rank a complex partly on how thinly its size class trades. Narrower peer sets
    (자치구 × 면적군) were rejected — several of those cells hold fewer than 20 combinations.
    """
    table = axis_inputs(quarterly, reference, partial)
    if table.empty:
        return {}
    recovery_ratios = table["recovery_ratio"].dropna()
    recovery_max = float(recovery_ratios.max()) if len(recovery_ratios) else None
    scores = {}
    for area_group, peers in table.groupby("area_group", observed=True):
        # A complex that never fell below its running peak tops both axes: there was no
        # decline to defend against, and it currently sits at its own high.
        declined = peers.loc[~peers["gap"].isna()]
        declined_ids = set(declined.complex_id)
        totals, axes = {}, {}
        for row in peers.itertuples(index=False):
            defense = 100.0 if pd.isna(row.gap) else rank_percentile(declined["mdd"], row.mdd)
            recovery = 100.0 if pd.isna(row.gap) else rank_percentile(declined["gap"], row.gap)
            totals[row.complex_id] = (defense * DEFENSE_WEIGHT + recovery * RECOVERY_WEIGHT) / 100
            axes[row.complex_id] = (defense, recovery)
        ranked = pd.Series(totals)
        # Both the grade and the stated rank read against the same area group, so the note
        # on the card describes the population the score was actually computed from.
        positions = ranked.rank(ascending=False, method="min").astype(int)
        for complex_id, total in ranked.items():
            defense, recovery = axes[complex_id]
            percentile = rank_percentile(ranked, total)
            scores[(complex_id, area_group)] = {
                "grade": grade_of(percentile),
                "total": round(total),
                "defense": round(defense * DEFENSE_WEIGHT / 100),
                "recovery": round(recovery * RECOVERY_WEIGHT / 100),
                "rank": int(positions[complex_id]),
                "peers": len(ranked),
                "defense_percentile": defense,
                "recovery_percentile": recovery,
                "percentile": percentile,
                "axis_peers": len(declined),
                "no_decline": complex_id not in declined_ids,
                "recovery_max": recovery_max,
                "recovery_sample_count": len(recovery_ratios),
            }
    return scores
