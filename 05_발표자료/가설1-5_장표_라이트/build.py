"""가설 1~5 EDA 시각화(라이트 테마)와 장표용 숫자 텍스트."""
from pathlib import Path
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

SPEC = importlib.util.spec_from_file_location(
    "verified", ROOT / "05_발표자료/가설1-5_EDA_발표자료/build.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)

INK = "#172033"
MUTED = "#475569"
FAINT = "#64748b"
GRID = "#e2e8f0"
EDGE = "#94a3b8"
BLUE = "#2563eb"
ORANGE = "#d95f02"
GREEN = "#00856a"
GREY = "#94a3b8"

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": EDGE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
    "svg.fonttype": "path",
})


def canvas(title, subtitle):
    fig = plt.figure(figsize=(13.4, 7.5), dpi=180)
    fig.text(0.055, 0.945, title, fontsize=19, fontweight="bold", color=INK)
    fig.text(0.056, 0.898, subtitle, fontsize=11, color=MUTED)
    return fig


def close(fig, stem, message, note):
    fig.text(0.055, 0.088, message, fontsize=13.5, fontweight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f1f5f9", edgecolor=GRID))
    fig.text(0.055, 0.042, note, fontsize=9.2, color=FAINT)
    fig.text(0.055, 0.016,
             "원천: 아파트_거래_DATA_대시보드_실데이터.csv · 2017Q1~2026Q2 · 관찰된 연관성에 대한 EDA",
             fontsize=8.5, color=FAINT)
    fig.savefig(FIG / f"{stem}.png", dpi=180)
    fig.savefig(FIG / f"{stem}.svg")
    plt.close(fig)


def panel(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def h1(d):
    """같은 자치구·면적군 안의 분포와 비슷한 낙폭 3개 사례."""
    n = d[d.district.eq("노원구")]
    names = ["보람아파트", "건영3", "한진한화그랑빌"]
    focus = n[n.name.isin(names)].set_index("name").loc[names]
    colors = {"보람아파트": ORANGE, "건영3": BLUE, "한진한화그랑빌": GREEN}

    fig = canvas("가설 1. 같은 자치구·같은 면적군에서도 단지별 하락과 회복은 달랐다",
                 f"노원구 {len(n)}개 단지 · 전용 60㎡ 이상 85㎡ 미만 · 2026Q2 동일시점")
    ax, bx = fig.subplots(1, 2, gridspec_kw=dict(
        left=.075, right=.965, top=.83, bottom=.20, wspace=.24, width_ratios=[1.1, 0.9]))

    ax.scatter(n.mdd, n.recovery, s=32, color=GREY, alpha=.60,
               edgecolors="white", linewidths=.35, label=f"노원구 {len(n)}개 단지")
    ax.axhline(100, color=FAINT, ls="--", lw=1.4, label="MDD 기준 고점 회복 100%")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylim(-4, n.recovery.max() * 1.55)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(-35.5, 1.5)
    ax.set_xlabel("MDD (%)   ← 더 큰 하락                 더 작은 하락 →")
    ax.set_ylabel("하락회복률 (%) · 100% 초과 구간 로그 눈금")
    ax.set_title("동일 조건 안에서도 넓게 분포", loc="left", fontsize=13.5, fontweight="bold")
    for name in names:
        r = focus.loc[name]
        ax.scatter(r.mdd, r.recovery, s=125, color=colors[name],
                   edgecolors="white", linewidths=1.2, zorder=5)
    ax.legend(loc="upper left", fontsize=9.5)
    panel(ax, "both")

    bars = bx.bar(names, focus.recovery, color=[colors[x] for x in names], width=.62, zorder=3)
    bx.bar_label(bars, labels=[f"{v:.1f}%" for v in focus.recovery],
                 padding=5, fontsize=12, fontweight="bold")
    for bar, mdd in zip(bars, focus.mdd):
        bx.text(bar.get_x() + bar.get_width() / 2, 7, f"MDD\n{mdd:.1f}%", ha="center",
                va="bottom", color="white", fontsize=10, fontweight="bold")
    bx.axhline(100, color=FAINT, ls="--", lw=1.4)
    bx.set_ylim(0, 165)
    bx.set_ylabel("하락회복률 (%)")
    bx.set_xticks(range(3), ["보람아파트\n상계동", "건영3\n중계동", "한진한화그랑빌\n월계동"])
    bx.set_title("비슷한 하락폭도 회복 결과는 다름", loc="left", fontsize=13.5, fontweight="bold")
    panel(bx)

    gap = float(focus.recovery.iloc[2] - focus.recovery.iloc[0])
    close(fig, "01_가설1_동일조건_분포와_사례",
          f"약 30% 하락한 두 단지의 하락회복률은 {gap:.1f}%p 차이였다.",
          f"전체 {len(n)}개 단지: MDD 중앙값 {n.mdd.median():.1f}%"
          f"(25~75백분위 {n.mdd.quantile(.25):.1f}~{n.mdd.quantile(.75):.1f}%), "
          f"회복률 중앙값 {n.recovery.median():.1f}%"
          f"({n.recovery.quantile(.25):.1f}~{n.recovery.quantile(.75):.1f}%) · 사례의 저점 시기는 서로 다름")

    case = focus.reset_index()[["name", "dong", "mdd", "recovery", "peak_q", "trough_q"]]
    case.columns = ["단지", "법정동", "MDD(%)", "하락회복률(%)", "고점분기", "저점분기"]
    stat = pd.DataFrame([[
        len(n), n.mdd.median(), n.mdd.quantile(.25), n.mdd.quantile(.75),
        n.recovery.median(), n.recovery.quantile(.25), n.recovery.quantile(.75)]],
        columns=["단지수", "MDD중앙값", "MDD_Q1", "MDD_Q3", "회복률중앙값", "회복률_Q1", "회복률_Q3"])
    return case, stat, gap


def h2(s):
    rows = []
    for label, col in [("Q2 거래량", "basis_count"), ("관측분기당 거래량", "activity"),
                       ("누적 거래량", "trades")]:
        for target, y in [("가격 방어력", s.mdd), ("하락회복률", s.recovery)]:
            rows.append([label, target, V.corr(s, s[col], y),
                         V.corr(s, s[col], y, ["qualified_quarters"])])
    tb = pd.DataFrame(rows, columns=["거래지표", "결과지표", "자치구내상관", "유효분기수보정상관"])

    fig = canvas("가설 2. 거래가 활발한 단지일수록 잘 방어하고 더 회복했을까?",
                 f"같은 자치구 내 순위로 비교 · 60~85㎡ · MDD 절대폭 5% 이상 {len(s)}개")
    axs = fig.subplots(1, 2, gridspec_kw=dict(left=.185, right=.965, top=.80, bottom=.22, wspace=.30))
    for ax, (target, color) in zip(axs, [("가격 방어력", BLUE), ("하락회복률", ORANGE)]):
        t = tb[tb.결과지표.eq(target)]
        for j, r in enumerate(t.itertuples(index=False, name=None)):
            ax.plot([r[2], r[3]], [j, j], color=EDGE, lw=2, zorder=2)
            ax.scatter(r[2], j, s=85, facecolor="#ffffff", edgecolor=EDGE, linewidths=1.6, zorder=3)
            ax.scatter(r[3], j, s=95, color=color, zorder=4)
            ax.text(r[2] - .028, j + .20, f"{r[2]:+.2f}", ha="right", fontsize=10, color=MUTED)
            ax.text(r[3] + .028, j - .15, f"{r[3]:+.2f}", ha="left", fontsize=11.5,
                    color=color, fontweight="bold")
        ax.set_yticks(range(3), t.거래지표)
        ax.set_ylim(2.6, -.6)
        ax.set_xlim(-.66, .26)
        ax.axvline(0, color=FAINT, ls="--", lw=1.2)
        ax.set_title(target, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("상관계수 (+가 가설이 예상한 방향)")
        panel(ax, "x")
    fig.text(0.185, .842, "○ 자치구 내 비교        ● 유효분기 수까지 보정", fontsize=10.5, color=MUTED)

    lo, hi = tb.유효분기수보정상관.min(), tb.유효분기수보정상관.max()
    close(fig, "02_가설2_거래유동성",
          f"보정 후 상관은 {lo:+.2f}~{hi:+.2f} · 거래 활발도와의 연관은 약했다.",
          "유효분기 수=거래 3건 이상인 분기 수. 거래가 많은 단지가 통계적으로 안정된 값을 갖는 효과를 걷어낸 뒤의 상관임.")
    return tb


def h3(s, col, labels, bins, title, subtitle_kind, stem, message):
    x = s.dropna(subset=[col]).copy()
    x["group"] = pd.cut(x[col], bins, labels=labels, right=False)
    # 기준선은 결측 행을 포함한 표본 전체의 자치구 중앙값(검증된 build.py와 동일).
    for c in ["mdd", "recovery"]:
        x[c + "_rel"] = x[c] - x.district.map(s.groupby("district")[c].median())
    rows = []
    for name, g in x.groupby("group", observed=True):
        rows.append([str(name), len(g), g.mdd.median(), g.recovery.median(),
                     g.mdd_rel.median(), g.recovery_rel.median()])
    tb = pd.DataFrame(rows, columns=["구분", "n", "MDD중앙값(%)", "회복률중앙값(%)",
                                     "방어력_자치구대비(%p)", "회복률_자치구대비(%p)"])

    fig = canvas(title, f"같은 자치구 중앙값 대비 차이 · 60~85㎡ · MDD 절대폭 5% 이상 · "
                        f"유효 표본 {len(x)}개 (결측 {len(s) - len(x)}개)")
    axs = fig.subplots(1, 2, gridspec_kw=dict(left=.075, right=.965, top=.81, bottom=.22, wspace=.24))
    for ax, c, color, ylab in [
            (axs[0], "방어력_자치구대비(%p)", BLUE, "가격 방어력 차이 (%p) · +가 덜 하락"),
            (axs[1], "회복률_자치구대비(%p)", ORANGE, "하락회복률 차이 (%p) · +가 더 회복")]:
        ax.bar(tb.구분, tb[c], color=color, width=.55, zorder=3)
        span = max(tb[c].abs().max(), 1e-9)
        pad = span * 0.06
        for i, v in enumerate(tb[c]):
            ax.text(i, v + pad * (1 if v >= 0 else -1), f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=15, fontweight="bold")
        ax.set_xticks(range(len(tb)), [f"{a}\n(n={b})" for a, b in zip(tb.구분, tb.n)])
        ax.set_ylim(-span * 1.32, span * 1.32)
        ax.axhline(0, color=FAINT, lw=1.2)
        ax.set_title(ylab, loc="left", fontsize=12.5)
        panel(ax)
    close(fig, stem, message,
          f"+는 같은 자치구 중앙값보다 좋은 값. {subtitle_kind}별 집단 중앙값 비교이며 인과 효과나 유의성 검정이 아님.")
    return tb


def h4(s):
    rows = []
    for kind in ["지하철", "학교", "공원", "대규모점포"]:
        v = s.dropna(subset=[kind])
        near = -v[kind]
        rows.append([kind, len(v), len(v) / len(s) * 100,
                     V.corr(v, near, v.mdd, ["qualified_quarters"]),
                     V.corr(v, near, v.recovery, ["qualified_quarters"])])
    tb = pd.DataFrame(rows, columns=["시설", "n", "유효비율(%)", "가격방어력상관", "하락회복률상관"])

    fig = canvas("가설 4. 교통·생활시설이 가까운 단지일수록 방어·회복이 좋았을까?",
                 f"같은 자치구 내 접근성 순위 + 유효분기 수 보정 · 60~85㎡ · 전체 {len(s)}개 중 유효 거리 표본")
    ax = fig.add_axes([.20, .22, .54, .55])
    arr = tb[["가격방어력상관", "하락회복률상관"]].to_numpy()
    ax.imshow(arr, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:+.3f}", ha="center", va="center",
                    fontsize=22, color=INK, fontweight="bold")
    ax.set_xticks([0, 1], ["가격 방어력", "하락회복률"])
    ax.set_yticks(range(4), tb.시설)
    ax.tick_params(labelsize=12.5, colors=INK)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(EDGE)
    for i, r in tb.iterrows():
        fig.text(.775, .715 - i * .1375, f"n={r['n']}\n유효 {r['유효비율(%)']:.1f}%",
                 fontsize=10.5, color=MUTED, va="center")
    fig.text(.20, .805, "+는 가까울수록 좋은 결과 · 0은 순위상 연관이 거의 없음 · 색 범위 -1~+1",
             fontsize=10, color=MUTED)
    close(fig, "05_가설4_입지환경",
          f"관측된 절대 상관은 모두 {np.abs(arr).max():.3f} 이하로 작았다.",
          "입지는 최근접 시설까지의 거리만 측정. 노선·학군·시설 품질은 담지 못하므로 입지가 무관하다는 뜻은 아님.")
    return tb


def h5(s):
    summary, scores, pred = V.model(s)
    fig = canvas("가설 5. 거래량·단지 특성·입지를 함께 고려해도 차이가 남았을까?",
                 f"같은 자치구를 고려한 Ridge 회귀 · 5분할 × 5회 반복 교차검증 · {len(s)}개 · 외부 검증 아님")
    axs = fig.subplots(1, 2, gridspec_kw=dict(left=.085, right=.965, top=.79, bottom=.30, wspace=.25))
    for ax, target, title in zip(axs, ["MDD 절대폭", "log(1+하락회복률)"],
                                 ["MDD 절대폭 예측", "하락회복률 예측 · 로그 변환"]):
        t = summary[summary.목표.eq(target)].set_index("모형").loc[["기준모형", "종합모형"]]
        ax.bar(["기준모형", "종합모형"], t.R2, color=[GREY, BLUE], width=.48, zorder=3)
        for i, v in enumerate(t.R2):
            ax.text(i, v + .035, f"{v:.3f}", ha="center", fontsize=19, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontsize=12.5)
        ax.set_ylabel("교차검증 R² · 높을수록 예측 오차가 작음")
        panel(ax)
        r = t.loc["종합모형"]
        ax.text(.5, -.235, f"종합모형 평균절대오차 {r.MAE:.1f}%p   ·   중앙절대오차 {r.MedAE:.1f}%p",
                ha="center", transform=ax.transAxes, fontsize=11.5, color=ORANGE, fontweight="bold")
    base_mae = float(summary[summary.모형.eq("기준모형") & summary.목표.eq("log(1+하락회복률)")].MAE.iloc[0])
    close(fig, "06_가설5_종합모형_남는오차",
          "특성을 추가해도 R² 개선은 작았고, 회복률의 평균절대오차는 줄지 않았다.",
          f"기준=자치구+관측이력 / 종합=기준+거래량·세대수·연식·세대당 주차·시설 거리. "
          f"회복률 기준모형 평균절대오차 {base_mae:.1f}%p. R²의 회복률 값은 로그 척도.")
    return summary, pred


def report(d, s, case, stat, gap, h2t, hh, age, loc, summary, pred):
    def row(t, i, c):
        return t.iloc[i][c]

    mdl = summary.set_index(["모형", "목표"])
    m_mdd_b = mdl.loc[("기준모형", "MDD 절대폭")]
    m_mdd_f = mdl.loc[("종합모형", "MDD 절대폭")]
    m_rec_b = mdl.loc[("기준모형", "log(1+하락회복률)")]
    m_rec_f = mdl.loc[("종합모형", "log(1+하락회복률)")]

    text = f"""# 가설 1~5 · 장표에 바로 넣는 숫자

공통 조건 — 전용 60㎡ 이상 85㎡ 미만 · 기준시점 2026Q2 · 거래 3건 이상 완결 분기 4개 이상.
가설 1은 하락이 관측된 {len(d)}개 단지, 가설 2~5는 MDD 절대폭 5% 이상 **{len(s)}개** 표본.
아래는 모두 관찰된 연관성에 대한 EDA이며 인과 규명이나 미래 예측 검증이 아닙니다.

---

## 장표 1 — 같은 자치구·같은 면적군에서도 하락과 회복은 달랐다

![가설1](figures/01_가설1_동일조건_분포와_사례.png)

**노원구 {int(stat.단지수.iloc[0])}개 단지 전체 분포**

| 지표 | 중앙값 | 25백분위 | 75백분위 |
| --- | --- | --- | --- |
| MDD (%) | {stat.MDD중앙값.iloc[0]:.1f}% | {stat.MDD_Q1.iloc[0]:.1f}% | {stat.MDD_Q3.iloc[0]:.1f}% |
| 하락회복률 (%) | {stat.회복률중앙값.iloc[0]:.1f}% | {stat.회복률_Q1.iloc[0]:.1f}% | {stat.회복률_Q3.iloc[0]:.1f}% |

**비슷한 낙폭, 다른 결과 (3개 사례)**

| 단지 | 법정동 | MDD | 하락회복률 | 고점 | 저점 |
| --- | --- | --- | --- | --- | --- |
| {row(case,0,'단지')} | {row(case,0,'법정동')} | {row(case,0,'MDD(%)'):.1f}% | {row(case,0,'하락회복률(%)'):.1f}% | {row(case,0,'고점분기')} | {row(case,0,'저점분기')} |
| {row(case,1,'단지')} | {row(case,1,'법정동')} | {row(case,1,'MDD(%)'):.1f}% | {row(case,1,'하락회복률(%)'):.1f}% | {row(case,1,'고점분기')} | {row(case,1,'저점분기')} |
| {row(case,2,'단지')} | {row(case,2,'법정동')} | {row(case,2,'MDD(%)'):.1f}% | {row(case,2,'하락회복률(%)'):.1f}% | {row(case,2,'고점분기')} | {row(case,2,'저점분기')} |

- 낙폭이 사실상 같은 두 단지({row(case,0,'단지')} {row(case,0,'MDD(%)'):.1f}% / {row(case,2,'단지')} {row(case,2,'MDD(%)'):.1f}%)의 회복률 격차 **{gap:.1f}%p**.
- 회복률 사분위 범위가 {stat.회복률_Q1.iloc[0]:.1f}~{stat.회복률_Q3.iloc[0]:.1f}%로 넓게 퍼져 있음.
- 단, 저점 시기가 서로 달라 회복에 주어진 기간도 다름.

> 전환 멘트 — "같은 자치구와 같은 면적군인데도 왜 이런 차이가 발생했을까요? 저희는 그 이유를 거래 유동성, 단지 특성, 입지환경의 세 가지 측면에서 살펴봤습니다."

---

## 장표 2 — 거래량만으로는 방어와 회복을 설명하기 어려웠다

![가설2](figures/02_가설2_거래유동성.png)

| 거래지표 | 결과지표 | 자치구 내 상관 | 유효분기 수 보정 후 |
| --- | --- | --- | --- |
""" + "\n".join(
        f"| {r.거래지표} | {r.결과지표} | {r.자치구내상관:+.3f} | **{r.유효분기수보정상관:+.3f}** |"
        for r in h2t.itertuples()) + f"""

- 보정 전 방어력 상관이 최대 {h2t[h2t.결과지표.eq('가격 방어력')].자치구내상관.min():+.2f}까지 나오지만, 이는 거래가 많은 단지일수록 값이 안정되는 효과가 섞인 결과.
- 유효분기 수를 보정하면 상관은 **{h2t.유효분기수보정상관.min():+.3f} ~ {h2t.유효분기수보정상관.max():+.3f}** 로 축소.
- 방어력과 회복률의 부호도 서로 엇갈려, "거래가 활발하면 둘 다 좋다"는 예상은 일관되게 확인되지 않음.

> 전환 멘트 — "거래 활발도만으로는 차이를 설명하기 어려웠습니다. 다음으로 단지의 규모와 연식을 봤습니다."

---

## 장표 3-1 — 대단지의 방어력과 회복률은 같은 방향이 아니었다

![가설3-1](figures/03_가설3-1_세대수.png)

| 세대수 구간 | n | MDD 중앙값 | 회복률 중앙값 | 방어력(자치구 대비) | 회복률(자치구 대비) |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(
        f"| {r.구분} | {r.n} | {getattr(r, '_3'):.1f}% | {getattr(r, '_4'):.1f}% | "
        f"**{getattr(r, '_5'):+.1f}%p** | **{getattr(r, '_6'):+.1f}%p** |"
        for r in hh.itertuples()) + f"""

- 1,000세대 이상: 방어력 **{hh.iloc[-1, 4]:+.1f}%p**, 회복률 **{hh.iloc[-1, 5]:+.1f}%p** — 더 잘 버티지는 않았고 회복만 소폭 우위.
- 300세대 미만: 방어력 **{hh.iloc[0, 4]:+.1f}%p**, 회복률 **{hh.iloc[0, 5]:+.1f}%p** — 방향이 정반대.
- '대단지이면 덜 하락하고 더 회복한다'고 묶어서 판단하기 어려움.

## 장표 3-2 — 신축일수록 두 지표가 일관되게 좋지는 않았다

![가설3-2](figures/04_가설3-2_연식.png)

| 연식 구간 | n | MDD 중앙값 | 회복률 중앙값 | 방어력(자치구 대비) | 회복률(자치구 대비) |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(
        f"| {r.구분} | {r.n} | {getattr(r, '_3'):.1f}% | {getattr(r, '_4'):.1f}% | "
        f"**{getattr(r, '_5'):+.1f}%p** | **{getattr(r, '_6'):+.1f}%p** |"
        for r in age.itertuples()) + f"""

- 10년 이하: 방어력 **{age.iloc[0, 4]:+.1f}%p**, 회복률 **{age.iloc[0, 5]:+.1f}%p**.
- 연식이 오래될수록 나빠지는 단조 관계는 나타나지 않음(11~20년 회복률 {age.iloc[1, 5]:+.1f}%p, 21년 이상 {age.iloc[2, 5]:+.1f}%p).
- 연식 순서에 따른 일관된 방어·회복 우위는 확인되지 않음.

> 전환 멘트 — "규모와 연식에 이어, 같은 자치구 안에서도 달라지는 시설 접근성을 봤습니다."

---

## 장표 4 — 최근접 시설 거리와의 연관은 작았다

![가설4](figures/05_가설4_입지환경.png)

| 시설 | 유효 n | 유효비율 | 가격 방어력 상관 | 하락회복률 상관 |
| --- | --- | --- | --- | --- |
""" + "\n".join(
        f"| {r.시설} | {r.n} | {getattr(r, '_3'):.1f}% | {r.가격방어력상관:+.3f} | {r.하락회복률상관:+.3f} |"
        for r in loc.itertuples()) + f"""

- 자치구와 유효분기 수를 함께 보정한 절대 상관 최대값 **{loc[['가격방어력상관','하락회복률상관']].abs().to_numpy().max():.3f}**.
- 거리 변수만으로는 단지별 차이를 설명하기 어려움.
- 입지가 중요하지 않다는 결론은 아님 — 거리는 노선·학군·시설 품질을 담지 못함.

> 전환 멘트 — "마지막으로 변수를 하나씩 보는 단계를 넘어, 여러 특성을 함께 넣어봤습니다."

---

## 장표 5 — 함께 고려해도 단지별 차이는 남았다

![가설5](figures/06_가설5_종합모형_남는오차.png)

| 목표 | 기준모형 R² | 종합모형 R² | 개선폭 | 종합모형 평균절대오차 | 중앙절대오차 |
| --- | --- | --- | --- | --- | --- |
| MDD 절대폭 | {m_mdd_b.R2:.3f} | {m_mdd_f.R2:.3f} | {m_mdd_f.R2 - m_mdd_b.R2:+.3f} | {m_mdd_f.MAE:.1f}%p | {m_mdd_f.MedAE:.1f}%p |
| 하락회복률(로그) | {m_rec_b.R2:.3f} | {m_rec_f.R2:.3f} | {m_rec_f.R2 - m_rec_b.R2:+.3f} | {m_rec_f.MAE:.1f}%p | {m_rec_f.MedAE:.1f}%p |

- 기준모형 = 자치구 + 관측이력 / 종합모형 = 기준 + 거래량·세대수·연식·세대당 주차·시설 거리.
- 특성을 모두 넣어도 R² 개선은 MDD **{m_mdd_f.R2 - m_mdd_b.R2:+.3f}**, 회복률 **{m_rec_f.R2 - m_rec_b.R2:+.3f}** 에 그침.
- 회복률의 원래 단위 평균절대오차는 기준 {m_rec_b.MAE:.1f}%p → 종합 {m_rec_f.MAE:.1f}%p 로 **줄지 않음**.
- 종합모형으로도 MDD 예측이 **{float((pred.MDD잔차.abs() > 5).mean()) * 100:.0f}%** 의 단지에서 5%p 넘게 빗나갔고, 회복률은 **{float((pred.회복률잔차.abs() > 50).mean()) * 100:.0f}%** 의 단지에서 50%p 넘게 빗나감.
- 잔차에는 누락 요인·거래 잡음·모형 한계가 함께 포함됨. 단지 고유의 힘을 입증한 것은 아님.

---

## 마무리 멘트

> 같은 자치구·같은 면적군에서도 단지별 하락과 회복은 달랐습니다. 거래 유동성, 단지 특성, 입지환경을 차례로 살펴봤지만 몇 가지 특성만으로 모든 단지를 설명하기는 어려웠습니다. 그래서 자치구 평균만 보는 대신, 관심 단지의 가격 흐름과 MDD·하락회복률, 단지 특성을 직접 비교할 수 있는 서비스를 만들었습니다.
"""
    (OUT / "가설1-5_장표숫자.md").write_text(text, encoding="utf-8")


def main():
    d = V.load()
    s = d[d.loss.ge(5)].copy()
    print(f"검증 표본 {len(d)} / 가설2~5 표본 {len(s)}", flush=True)

    case, stat, gap = h1(d)
    h2t = h2(s)
    hh = h3(s, "households", ["300세대 미만", "300–999세대", "1,000세대 이상"],
            [0, 300, 1000, np.inf], "가설 3-1. 세대수가 많은 대단지일수록 방어·회복이 좋았을까?",
            "세대수", "03_가설3-1_세대수",
            "대단지는 방어력이 낮고 회복률은 소폭 높았다. 두 지표가 같은 방향으로 움직이지 않았다.")
    age = h3(s, "age", ["10년 이하", "11–20년", "21년 이상"],
             [0, 11, 21, np.inf], "가설 3-2. 상대적으로 신축인 단지일수록 방어·회복이 좋았을까?",
             "연식", "04_가설3-2_연식",
             "연식별 방어력 차이는 작고, 회복률은 연식 순서대로 일관되게 높아지지 않았다.")
    loc = h4(s)
    summary, pred = h5(s)

    for name, t in [("H1_사례", case), ("H1_노원구_분포", stat), ("H2_거래유동성_상관", h2t),
                    ("H3-1_세대수", hh), ("H3-2_연식", age), ("H4_입지환경_상관", loc),
                    ("H5_교차검증_요약", summary)]:
        t.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")

    report(d, s, case, stat, gap, h2t, hh, age, loc, summary, pred)
    (OUT / "검증_요약.json").write_text(json.dumps({
        "source_sha256": V.RESULTS.get("source_sha256"),
        "n_validated": len(d), "n_primary": len(s),
        "h5": summary.to_dict("records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("figures 6종 + 장표숫자 markdown 저장 완료", flush=True)


if __name__ == "__main__":
    main()
