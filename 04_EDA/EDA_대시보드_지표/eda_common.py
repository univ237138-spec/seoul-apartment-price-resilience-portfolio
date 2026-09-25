"""EDA 스크립트 공통 설정: 한글 폰트, 팔레트, 저장 헬퍼, 대시보드 지표 재현.

팔레트는 dataviz 스킬의 검증된 기준 팔레트를 그대로 사용한다.
- 범주형: 문서화된 8슬롯 고정 순서. 인접쌍 게이트를 모두 통과한 순서이므로 순환하지 않고 앞에서부터 배정한다.
- 산점도처럼 모든 쌍이 동시에 비교되는 형태는 all-pairs 통과가 확인된 앞 3슬롯까지만 쓴다.
- 연속형은 파랑 단일 색상 램프, 양극성은 파랑↔빨강 + 회색 중간값.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "_cache"
FIGURES = Path(__file__).resolve().parent / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e3e2dd"

# 범주형 고정 순서 (blue, orange, aqua, yellow, magenta, green, violet, red)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# 산점도·소형 다중 등 all-pairs 형태에서 안전한 범위
CAT_ALLPAIRS = CAT[:3]

SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#0d366b", "#2a78d6", "#9ec5f4",
                                                        "#f0efec", "#f5a3a2", "#e34948", "#8f1f1f"])
STATUS = {"good": "#1baf7a", "warning": "#eda100", "serious": "#eb6834", "critical": "#e34948"}

# 회복률 해석 구간 (기획서 정의) — 낮을수록 위험
RECOVERY_BANDS = [(-np.inf, 50, "회복 지연", STATUS["critical"]),
                  (50, 80, "회복 진행", STATUS["serious"]),
                  (80, 100, "전고점 근접", STATUS["warning"]),
                  (100, np.inf, "전고점 회복·돌파", STATUS["good"])]

MIN_QUARTER_TRADES = 3
MIN_COMPLETE_QUARTERS = 4


def setup():
    mpl.rcParams.update({
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    })


def grid_y(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def grid_x(ax):
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, note=None):
    FIGURES.mkdir(parents=True, exist_ok=True)
    if note:
        fig.text(0.0, -0.015, note, fontsize=8, color=MUTED, ha="left", va="top")
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


def load(name):
    df = pd.read_parquet(CACHE / f"{name}.parquet")
    if "quarter" in df.columns:
        df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    return df


def meta():
    import json
    return json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))


def band_of(recovery):
    for low, high, label, _ in RECOVERY_BANDS:
        if low <= recovery < high:
            return label
    return "회복 지연"


def compute_metrics(quarterly: pd.DataFrame, reference: pd.Period) -> pd.DataFrame:
    """housing.metrics.build_snapshot의 지표를 단지×면적군 전체에 대해 한 번에 계산한다.

    build_snapshot은 화면에 띄울 한 조합만 다루므로, 지표 분포를 보려면 같은 규칙을
    모든 조합에 적용해야 한다. 규칙은 metrics.py와 동일하게 유지한다:
      · 완결 분기이고 분기 거래 3건 이상인 관측만 자격 표본(qualified)
      · 자격 표본 4개 이상 + 기준 분기 거래 3건 이상일 때만 지표 산출
      · 전고점은 저점 이전 구간의 최대값, 동일 낙폭이면 앞선 분기를 저점으로 택함
    """
    q = quarterly.loc[quarterly.quarter <= reference].sort_values(["complex_id", "area_group", "quarter"])
    current = q.loc[q.quarter == reference].set_index(["complex_id", "area_group"])
    qualified = q.loc[q.complete & q["count"].ge(MIN_QUARTER_TRADES)]

    rows = []
    for (cid, area), g in qualified.groupby(["complex_id", "area_group"], observed=True, sort=False):
        if len(g) < MIN_COMPLETE_QUARTERS or (cid, area) not in current.index:
            continue
        cur = current.loc[(cid, area)]
        cur_count = int(cur["count"])
        if cur_count < MIN_QUARTER_TRADES:
            continue
        prices = g.set_index("quarter")["price"]
        drawdown = (prices / prices.cummax() - 1) * 100
        trough_q = drawdown.idxmin()
        mdd = float(drawdown.loc[trough_q])
        price = float(cur["price"])
        row = {"complex_id": cid, "area_group": area, "price": price,
               "quarter_count": cur_count, "qualified_quarters": len(g),
               "observed_quarters": int((q.complex_id.eq(cid) & q.area_group.eq(area)).sum()),
               "total_trades": int(g["count"].sum()), "mdd": mdd,
               "first_quarter": str(g["quarter"].iloc[0]), "no_decline": bool(np.isclose(mdd, 0))}
        if not row["no_decline"]:
            peak_q = prices.loc[:trough_q].idxmax()
            peak, trough = float(prices.loc[peak_q]), float(prices.loc[trough_q])
            recovered = prices.loc[(prices.index > trough_q) & prices.ge(peak)]
            row.update(peak=peak, trough=trough, peak_quarter=str(peak_q), trough_quarter=str(trough_q),
                       gap=(price / peak - 1) * 100,
                       recovery=(price - trough) / (peak - trough) * 100,
                       recovery_quarters=(int(recovered.index[0].ordinal - trough_q.ordinal)
                                          if len(recovered) else np.nan),
                       recovered=bool(len(recovered)))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["band"] = out["recovery"].map(lambda v: band_of(v) if pd.notna(v) else "하락 구간 없음")
    return out


def eligibility(quarterly: pd.DataFrame, reference: pd.Period) -> pd.DataFrame:
    """모든 단지×면적군 조합이 대시보드 지표 조건을 어디에서 탈락하는지 표시한다."""
    q = quarterly.loc[quarterly.quarter <= reference]
    combos = q.groupby(["complex_id", "area_group"], observed=True)
    out = pd.DataFrame({"observed_quarters": combos.size(),
                        "trades": combos["count"].sum()})
    qual = q.loc[q.complete & q["count"].ge(MIN_QUARTER_TRADES)].groupby(
        ["complex_id", "area_group"], observed=True).size()
    out["qualified_quarters"] = qual.reindex(out.index).fillna(0).astype(int)
    cur = q.loc[q.quarter == reference].set_index(["complex_id", "area_group"])["count"]
    out["reference_count"] = cur.reindex(out.index).fillna(0).astype(int)
    out["has_reference"] = out["reference_count"] > 0
    out["enough_history"] = out["qualified_quarters"] >= MIN_COMPLETE_QUARTERS
    out["enough_reference"] = out["reference_count"] >= MIN_QUARTER_TRADES
    out["full_metrics"] = out["enough_history"] & out["enough_reference"]
    out["stage"] = np.select(
        [out["full_metrics"], out["enough_history"] & ~out["enough_reference"],
         ~out["enough_history"] & out["enough_reference"]],
        ["지표 전체 표시", "기준 분기 표본 부족", "장기 이력 부족"], "이력·기준 분기 모두 부족")
    return out.reset_index()


def dump(findings: dict, name: str):
    """numpy 스칼라가 섞여 있어도 저장되도록 float으로 낮춰 직렬화한다."""
    import json
    path = CACHE / f"{name}.json"
    path.write_text(json.dumps(findings, ensure_ascii=False, indent=2,
                              default=lambda o: float(o) if hasattr(o, "dtype") else str(o)),
                    encoding="utf-8")
    print(f"  findings -> {path.name}")
    return path
