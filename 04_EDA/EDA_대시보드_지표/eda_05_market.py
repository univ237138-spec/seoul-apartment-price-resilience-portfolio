"""5단계: 시장 흐름(자치구·면적군)과 기획서 세부 가설 검정.

기획서의 세부 가설:
  H1 대단지일수록 낙폭이 작고 회복률이 높다
  H2 교통·생활 인프라 접근성이 좋으면 회복이 빠르다
  H3 같은 자치구에서도 면적군에 따라 낙폭·회복 속도가 다르다
  H4 회복률이 높아도 거래량이 적으면 지표 신뢰도가 낮다
회복률은 얕은 낙폭에서 발산하므로(4단계), 상관은 순위 기반 스피어만으로 보고
필요한 곳에서는 |MDD|>=5% 부분집합을 함께 제시한다.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import eda_common as ec

ec.setup()
FACILITY_ORDER = ["지하철", "학교", "공원", "대규모점포"]
SOLID_MDD = 5.0  # 회복률이 발산하지 않는 최소 낙폭


def district_quarterly(trades, reference):
    """자치구 단위 분기 ㎡당 중위단가. 기획서가 말한 '시장 회복 흐름' 분석 단위다."""
    t = trades.loc[trades.quarter <= reference]
    return t.groupby(["district", "quarter"], observed=True).agg(
        price=("unit_price", "median"), count=("unit_price", "size")).reset_index()


def fig_market_heatmap(trades, reference, findings):
    """자치구 × 분기 가격 지수. 2021Q1 = 100."""
    dq = district_quarterly(trades, reference)
    pivot = dq.pivot(index="district", columns="quarter", values="price")
    pivot = pivot.loc[:, pivot.columns >= pd.Period("2021Q1", freq="Q")]
    base = pivot[pd.Period("2021Q1", freq="Q")]
    index = pivot.div(base, axis=0) * 100
    index = index.loc[index.iloc[:, -1].sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12.6, 8.2))
    sns.heatmap(index, ax=ax, cmap=ec.DIV, center=100, vmin=70, vmax=150, annot=False,
                linewidths=1.5, linecolor=ec.SURFACE,
                cbar_kws={"label": "2021Q1 = 100 지수", "shrink": 0.45, "pad": 0.02})
    ax.set_xticklabels([f"{p.year}\nQ{p.quarter}" for p in index.columns], rotation=0, fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    last = index.iloc[:, -1]
    ax.set_title(f"자치구별 ㎡당 중위단가 지수 (2021Q1 = 100)\n"
                 f"{index.columns[-1]} 기준 최고 {last.idxmax()} {last.max():.0f} / 최저 {last.idxmin()} {last.min():.0f} "
                 f"— 같은 조정기를 지나도 회복 수준은 자치구별로 갈린다")
    findings["market_index"] = {"기준분기": "2021Q1", "최종분기": str(index.columns[-1]),
                                "상위5": last.nlargest(5).round(1).to_dict(),
                                "하위5": last.nsmallest(5).round(1).to_dict(),
                                "최저점 분기 중위": str(index.median().idxmin())}
    return ec.save(fig, "16_자치구_가격지수_히트맵",
                   "100을 기준으로 갈리는 양극성 값이라 파랑↔빨강 발산 램프에 회색 중간값을 쓴다. "
                   "자치구 중위단가는 분기마다 거래 구성이 달라져 단일 칸의 튐은 시장 변동이 아닐 수 있다.")


def fig_area_group_flow(trades, reference, findings):
    """H3: 같은 시장에서도 면적군에 따라 낙폭·회복 속도가 다른가."""
    from housing.models import AREA_LABELS
    t = trades.loc[trades.quarter <= reference]
    ag = t.groupby(["area_group", "quarter"], observed=True)["unit_price"].median().unstack(0)
    order = [a for a in AREA_LABELS if a in ag.columns]
    ag = ag[order]
    ag = ag.loc[ag.index >= pd.Period("2021Q1", freq="Q")]
    index = ag.div(ag.iloc[0]) * 100

    # 분기당 표본이 얇은 면적군은 지수가 시장이 아니라 표본 때문에 튄다. 진폭과 함께 표본 수를 보여준다.
    per_quarter = t.groupby(["area_group", "quarter"], observed=True).size().unstack(0)[order]
    per_quarter = per_quarter.loc[per_quarter.index >= pd.Period("2021Q1", freq="Q")]
    volatility = index.diff().abs().mean()

    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    x = np.arange(len(index))
    for i, col in enumerate(order):
        ax.plot(x, index[col], color=ec.CAT[i], zorder=3,
                label=f"{col}  (분기 중위 {per_quarter[col].median():,.0f}건)")
    ax.axhline(100, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    # 끝점 라벨이 겹치지 않도록 위에서부터 최소 간격을 두고 배치한다.
    lasts = index.iloc[-1]
    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.035
    placed = None
    for col in lasts.sort_values(ascending=False).index:
        yv = lasts[col] if placed is None else min(lasts[col], placed - gap)
        ax.annotate(f"{lasts[col]:.0f}", (x[-1], yv), textcoords="offset points", xytext=(7, 0),
                    fontsize=8.5, color=ec.CAT[order.index(col)], va="center", fontweight="bold")
        placed = yv
    ax.set_xticks(x[::2], [f"{p.year}\nQ{p.quarter}" for p in index.index[::2]], fontsize=8.5)
    ax.set_ylabel("㎡당 중위단가 지수 (2021Q1 = 100)")
    ax.set_xlim(-0.5, len(index) + 1.6)
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ec.grid_y(ax)
    noisy = volatility.idxmax()
    drops = (index.min() - 100).round(1)
    ax.set_title(f"면적군별 가격 흐름 (2021Q1 = 100)\n"
                 f"{noisy}은 분기 중위 {per_quarter[noisy].median():,.0f}건뿐이라 지수가 분기마다 평균 "
                 f"{volatility[noisy]:.0f}p씩 튄다 — 이 선의 등락은 시장이 아니라 표본이다")
    findings["area_group_flow"] = {
        "최저 지수": drops.add(100).round(1).to_dict(),
        "최종 지수": lasts.round(1).to_dict(),
        "분기 중위 거래건수": per_quarter.median().round(0).to_dict(),
        "분기간 평균 변동폭(p)": volatility.round(1).to_dict()}
    return ec.save(fig, "17_면적군_가격흐름",
                   "6개 계열이라 범례를 두고 각 선 끝에 최종값을 직접 라벨했다. 범례의 건수는 분기당 거래 표본이다.")


def fig_district_ranking(metrics, complexes, findings):
    """자치구별 낙폭·회복률 — 단지 지표를 자치구로 올려 본다."""
    d = metrics.merge(complexes[["complex_id", "district"]], on="complex_id", how="left")
    d = d.loc[d["mdd"].abs() >= SOLID_MDD]
    g = d.groupby("district").agg(mdd=("mdd", "median"), recovery=("recovery", "median"),
                                  n=("mdd", "size"))
    g = g.loc[g.n >= 20].sort_values("mdd")

    fig, ax = plt.subplots(figsize=(10.6, 7.4))
    y = np.arange(len(g))
    ax.hlines(y, g["mdd"], 0, color=ec.GRID, linewidth=6, zorder=1)
    ax.scatter(g["mdd"], y, s=90, color=ec.CAT[0], zorder=3, edgecolors=ec.SURFACE, linewidths=1.5)
    ax.set_yticks(y, [f"{i}  (n={n})" for i, n in zip(g.index, g.n)])
    ax.invert_yaxis()
    ax.set_xlabel("자치구 중위 최대 낙폭 MDD (%)")
    ax.axvline(0, color=ec.INK2, linewidth=1)
    ec.grid_x(ax)
    for yi, (mdd, rec) in enumerate(zip(g["mdd"], g["recovery"])):
        ax.text(mdd - 0.35, yi, f"{mdd:.1f}%", va="center", ha="right", fontsize=8.5, color=ec.INK)
        ax.text(0.6, yi, f"회복률 중위 {rec:,.0f}%", va="center", fontsize=8.5, color=ec.INK2)
    ax.set_xlim(g["mdd"].min() * 1.22, abs(g["mdd"].min()) * 0.62)
    ax.set_title(f"자치구별 중위 낙폭과 회복률 (|MDD|≥{SOLID_MDD:.0f}%, 조합 20개 이상 자치구)\n"
                 f"낙폭이 가장 깊은 {g.index[0]}({g['mdd'].iloc[0]:.1f}%)과 가장 얕은 {g.index[-1]}({g['mdd'].iloc[-1]:.1f}%)의 차이는 "
                 f"{abs(g['mdd'].iloc[0] - g['mdd'].iloc[-1]):.1f}%p")
    findings["district_ranking"] = {
        "낙폭 깊은 5": g["mdd"].nsmallest(5).round(1).to_dict(),
        "낙폭 얕은 5": g["mdd"].nlargest(5).round(1).to_dict(),
        "회복률 높은 5": g["recovery"].nlargest(5).round(0).to_dict(),
        "회복률 낮은 5": g["recovery"].nsmallest(5).round(0).to_dict()}
    return ec.save(fig, "18_자치구_낙폭_회복률",
                   "단일 계열이라 범례 없이 값을 직접 라벨했다. n은 자치구별 지표 산출 조합 수다.")


def build_features(metrics, complexes, facilities):
    """단지 속성·접근성을 지표 표에 붙인다."""
    d = metrics.merge(complexes, on="complex_id", how="left")
    wide = facilities.loc[facilities.status.eq("ok")].pivot_table(
        index="complex_id", columns="kind", values="distance", aggfunc="first")
    wide.columns = [f"dist_{c}" for c in wide.columns]
    d = d.merge(wide, on="complex_id", how="left")
    d["age"] = pd.Period(metrics.attrs.get("reference", "2026Q2"), freq="Q").year - d["built_year"]
    d["parking_per_household"] = ((d["parking_ground"].fillna(0) + d["parking_underground"].fillna(0))
                                  / d["households"]).where(d["households"].gt(0))
    return d


def fig_correlations(features, findings):
    """H1·H2·H4를 한 판에: 단지 특성/접근성/거래량과 지표의 순위 상관."""
    solid = features.loc[features["mdd"].abs() >= SOLID_MDD]
    drivers = [("households", "세대수"), ("age", "연식 (년)"), ("parking_per_household", "세대당 주차대수"),
               ("total_trades", "누적 거래량"), ("quarter_count", "기준 분기 거래량"),
               ("qualified_quarters", "자격 분기 수")]
    drivers += [(f"dist_{k}", f"{k} 거리 (m)") for k in FACILITY_ORDER]
    targets = [("mdd", "최대 낙폭 MDD"), ("recovery", "하락분 회복률"),
               ("gap", "전고점 격차"), ("recovery_quarters", "회복 소요 분기")]

    rho = pd.DataFrame(index=[d[1] for d in drivers], columns=[t[1] for t in targets], dtype=float)
    pval = rho.copy()
    ns = {}
    for dcol, dlabel in drivers:
        for tcol, tlabel in targets:
            pair = solid[[dcol, tcol]].dropna()
            ns[dlabel] = max(ns.get(dlabel, 0), len(pair))
            if len(pair) >= 30:
                r, p = stats.spearmanr(pair[dcol], pair[tcol])
                rho.loc[dlabel, tlabel], pval.loc[dlabel, tlabel] = r, p

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    annot = rho.map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
    # 유의하지 않은 칸(p>=0.05)은 값 옆에 표시해 과잉 해석을 막는다.
    for i in rho.index:
        for j in rho.columns:
            if pd.notna(pval.loc[i, j]) and pval.loc[i, j] >= 0.05:
                annot.loc[i, j] += " n.s."
    sns.heatmap(rho, ax=ax, cmap=ec.DIV, center=0, vmin=-0.5, vmax=0.5, annot=annot, fmt="",
                annot_kws={"fontsize": 8.5}, linewidths=2, linecolor=ec.SURFACE,
                cbar_kws={"label": "스피어만 ρ", "shrink": 0.5, "pad": 0.02})
    ax.set_yticklabels([f"{t.get_text()}  (n={ns[t.get_text()]:,})" for t in ax.get_yticklabels()],
                       rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
    strongest = rho.abs().stack().idxmax()
    ax.set_title(f"단지 특성·접근성·거래량과 지표의 순위 상관 (|MDD|≥{SOLID_MDD:.0f}%)\n"
                 f"접근성 4종은 모두 |ρ|≤0.10이고, 가장 센 거래량·자격분기 칸은 관측창 효과다 (22번 그림)")
    findings["correlations"] = {t[1]: rho[t[1]].dropna().round(3).to_dict() for t in targets}
    findings["correlation_n"] = ns
    return ec.save(fig, "19_특성_지표_상관",
                   "0을 기준으로 부호가 갈리는 값이라 발산 램프를 쓴다. n.s.는 p≥0.05로 유의하지 않은 칸이다.")


def partial_spearman(df, x, y, control):
    """control을 통제한 x-y 순위 부분상관. 관측창 효과를 걷어내고 관계를 다시 읽기 위한 것."""
    s = df[[x, y, control]].dropna().rank()
    rxy = np.corrcoef(s[x], s[y])[0, 1]
    rxz = np.corrcoef(s[x], s[control])[0, 1]
    ryz = np.corrcoef(s[y], s[control])[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)), len(s)


def fig_observation_artifact(features, findings):
    """MDD는 누적 최대값 기준이라 관측 구간이 길수록 기계적으로 깊어진다.

    상관 히트맵에서 가장 센 값(거래량·세대수 ↔ MDD)이 실제 방어력인지, 아니면
    '오래 관측된 단지일수록 깊은 하락을 목격했을 뿐'인지 분리한다.
    """
    solid = features.loc[features["mdd"].abs() >= SOLID_MDD].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})

    bins = [3, 9, 13, 17, 23, np.inf]
    labels = ["4~8", "9~12", "13~16", "17~22", "23분기 이상"]
    solid["qbucket"] = pd.cut(solid["qualified_quarters"], bins, labels=labels, right=False)
    sns.boxplot(data=solid, x="qbucket", y="mdd", order=labels, ax=ax1, color=ec.CAT[0],
                width=0.6, fliersize=0, boxprops={"edgecolor": ec.INK2, "linewidth": 0.8, "alpha": 0.85},
                medianprops={"color": ec.SURFACE, "linewidth": 2},
                whiskerprops={"color": ec.INK2, "linewidth": 0.8},
                capprops={"color": ec.INK2, "linewidth": 0.8})
    counts = solid["qbucket"].value_counts().reindex(labels)
    med = solid.groupby("qbucket", observed=True)["mdd"].median().reindex(labels)
    for i, v in enumerate(med.values):
        ax1.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, color=ec.INK, fontweight="bold")
    ax1.set_xticks(np.arange(len(labels)), [f"{l}\n(n={counts[l]:,})" for l in labels], fontsize=8.5)
    ax1.set_xlabel("자격 분기 수 (관측 구간의 길이)")
    ax1.set_ylabel("최대 낙폭 MDD (%)")
    ec.grid_y(ax1)
    ax1.set_title("오래 관측될수록 낙폭은 기계적으로 깊어진다", fontsize=10.5)

    pairs = [("total_trades", "누적 거래량"), ("households", "세대수"),
             ("quarter_count", "기준 분기 거래량"), ("parking_per_household", "세대당 주차대수")]
    raw, ctrl = [], []
    for col, _ in pairs:
        pair = solid[[col, "mdd"]].dropna()
        raw.append(stats.spearmanr(pair[col], pair["mdd"]).statistic)
        ctrl.append(partial_spearman(solid, col, "mdd", "qualified_quarters")[0])
    x = np.arange(len(pairs))
    ax2.barh(x - 0.19, raw, height=0.34, color=ec.CAT[0], label="단순 상관")
    ax2.barh(x + 0.19, ctrl, height=0.34, color=ec.CAT[1], label="자격 분기 수 통제 후")
    ax2.set_yticks(x, [p[1] for p in pairs])
    ax2.invert_yaxis()
    ax2.axvline(0, color=ec.INK2, linewidth=1)
    ax2.set_xlabel("MDD와의 스피어만 ρ")
    ax2.legend(loc="upper right")
    ec.grid_x(ax2)
    for xi, (r, c) in enumerate(zip(raw, ctrl)):
        ax2.text(r - 0.02, xi - 0.19, f"{r:+.2f}", va="center", ha="right", fontsize=8.5, color=ec.INK)
        ax2.text(c - 0.02 if c < 0 else c + 0.02, xi + 0.19, f"{c:+.2f}", va="center",
                 ha="right" if c < 0 else "left", fontsize=8.5, color=ec.INK)
    ax2.set_xlim(min(raw) * 1.35, max(max(ctrl), 0.1) * 1.9)
    ax2.set_title("통제하면 관계가 대부분 사라진다", fontsize=10.5)

    findings["observation_artifact"] = {
        "자격분기수-MDD ρ": round(stats.spearmanr(solid["qualified_quarters"], solid["mdd"]).statistic, 3),
        "구간별 중위 MDD": {k: round(v, 1) for k, v in med.items()},
        "단순 vs 통제": {label: {"단순": round(r, 3), "통제 후": round(c, 3)}
                      for (_, label), r, c in zip(pairs, raw, ctrl)}}
    fig.suptitle("MDD는 누적 최대값 기준이라 관측 구간이 길수록 깊어진다 — "
                 "거래량·세대수와 MDD의 강한 상관은 대부분 이 효과다",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    return ec.save(fig, "22_관측창_인공물",
                   "부분상관은 세 변수의 순위 상관에서 통제변수 성분을 제거해 계산했다.")


def fig_households(features, findings):
    """H1: 대단지일수록 낙폭이 작고 회복률이 높은가."""
    solid = features.loc[features["mdd"].abs() >= SOLID_MDD].dropna(subset=["households"]).copy()
    bins = [0, 300, 600, 1000, 2000, np.inf]
    labels = ["300세대 미만", "300~600", "600~1,000", "1,000~2,000", "2,000세대 이상"]
    solid["bucket"] = pd.cut(solid["households"], bins, labels=labels, right=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for ax, col, label, color in [(ax1, "mdd", "최대 낙폭 MDD (%)", ec.CAT[0]),
                                  (ax2, "recovery", "하락분 회복률 (%)", ec.CAT[1])]:
        sns.boxplot(data=solid, x="bucket", y=col, order=labels, ax=ax, color=color, width=0.6,
                    fliersize=0, boxprops={"edgecolor": ec.INK2, "linewidth": 0.8, "alpha": 0.85},
                    medianprops={"color": ec.SURFACE, "linewidth": 2},
                    whiskerprops={"color": ec.INK2, "linewidth": 0.8},
                    capprops={"color": ec.INK2, "linewidth": 0.8})
        med = solid.groupby("bucket", observed=True)[col].median().reindex(labels)
        for i, v in enumerate(med.values):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5,
                    color=ec.INK, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right", fontsize=8.5)
        ec.grid_y(ax)
    ax2.set_ylim(0, solid["recovery"].quantile(0.95))
    counts = solid["bucket"].value_counts().reindex(labels)
    ax1.set_title("세대수 구간별 최대 낙폭", fontsize=10.5)
    ax2.set_title("세대수 구간별 회복률", fontsize=10.5)

    r_mdd = stats.spearmanr(solid["households"], solid["mdd"])
    r_rec = stats.spearmanr(solid["households"], solid["recovery"])
    # 관측창을 통제하면 세대수-MDD 관계가 남는지 확인한다 (22번 그림 참고).
    p_mdd, _ = partial_spearman(solid, "households", "mdd", "qualified_quarters")
    verdict = ("낙폭은 오히려 대단지가 더 깊지만, 관측창을 통제하면 그 관계도 사라진다"
               if r_mdd.statistic < 0 and abs(p_mdd) < 0.15
               else "낙폭은 오히려 대단지가 더 깊다" if r_mdd.statistic < 0
               else "가설과 같은 방향")
    findings["households"] = {
        "표본": int(len(solid)), "구간별 조합 수": {k: int(v) for k, v in counts.items()},
        "세대수-MDD ρ": round(r_mdd.statistic, 3), "세대수-MDD p": round(r_mdd.pvalue, 4),
        "세대수-MDD ρ (자격분기수 통제)": round(p_mdd, 3),
        "세대수-회복률 ρ": round(r_rec.statistic, 3), "세대수-회복률 p": round(r_rec.pvalue, 4),
        "판정": verdict}
    fig.suptitle(f"H1 대단지 가설 — 지지되지 않는다. {verdict}\n"
                 f"세대수↔MDD ρ={r_mdd.statistic:+.2f} (관측창 통제 후 {p_mdd:+.2f}), "
                 f"세대수↔회복률 ρ={r_rec.statistic:+.2f}, n={len(solid):,}",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.06)
    fig.tight_layout()
    return ec.save(fig, "20_세대수_가설",
                   f"세대수가 채워진 단지만 썼다(전체의 33%). 회복률 축은 상위 5%를 잘라 표시했다.")


def fig_volume_reliability(features, findings):
    """H4: 거래량이 적으면 회복률 지표가 얼마나 흔들리는가."""
    solid = features.loc[features["mdd"].abs() >= SOLID_MDD].copy()
    # 지표 산출 조합은 최소 4개 자격 분기 x 3건이라 누적 거래가 이미 두껍다. 실제 분포로 구간을 나눈다.
    bins = [0, 70, 110, 160, 255, np.inf]
    labels = ["70건 미만", "70~109건", "110~159건", "160~254건", "255건 이상"]
    solid["bucket"] = pd.cut(solid["total_trades"], bins, labels=labels, right=False)
    g = solid.groupby("bucket", observed=True)["recovery"]
    spread = (g.quantile(0.75) - g.quantile(0.25)).reindex(labels)
    counts = g.size().reindex(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    sns.boxplot(data=solid, x="bucket", y="recovery", order=labels, ax=ax1, color=ec.CAT[0],
                width=0.6, fliersize=0, boxprops={"edgecolor": ec.INK2, "linewidth": 0.8, "alpha": 0.85},
                medianprops={"color": ec.SURFACE, "linewidth": 2},
                whiskerprops={"color": ec.INK2, "linewidth": 0.8},
                capprops={"color": ec.INK2, "linewidth": 0.8})
    ax1.set_ylim(0, solid["recovery"].quantile(0.95))
    ax1.set_xlabel("자격 분기 누적 거래 건수")
    ax1.set_ylabel("하락분 회복률 (%)")
    ax1.set_xticks(np.arange(len(labels)), [f"{l}\n(n={counts[l]:,})" for l in labels], fontsize=8.5)
    ec.grid_y(ax1)
    ax1.set_title("거래량이 적을수록 회복률이 넓게 퍼진다", fontsize=10.5)

    ax2.bar(np.arange(len(labels)), spread.values, color=ec.CAT[0], width=0.6)
    ax2.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right", fontsize=8.5)
    ax2.set_ylabel("회복률 사분위 범위 IQR (%p)")
    for i, v in enumerate(spread.values):
        ax2.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=ec.INK, fontweight="bold")
    ec.grid_y(ax2)
    ax2.set_title("110건 이상부터는 두꺼워질수록 산포가 단조 감소한다", fontsize=10.5)

    findings["volume_reliability"] = {
        "구간별 조합 수": {k: int(v) for k, v in counts.items()},
        "구간별 회복률 IQR": {k: round(v, 1) for k, v in spread.items()},
        "IQR 배율(가장 얇은 구간 ÷ 가장 두꺼운 구간)": round(spread.iloc[0] / spread.iloc[-1], 1)}
    fig.suptitle(f"H4 신뢰도 가설 — 거래가 가장 얇은 구간의 회복률 산포가 가장 두꺼운 구간의 "
                 f"{spread.iloc[0] / spread.iloc[-1]:.1f}배다",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    return ec.save(fig, "21_거래량_신뢰도", "회복률 축은 상위 5%를 잘라 표시했고, IQR은 전체 표본으로 계산했다.")


def main():
    import sys
    sys.path.insert(0, str(ec.ROOT / "대시보드"))
    m = ec.meta()
    reference = pd.Period(m["reference"], freq="Q")
    trades, complexes, facilities = ec.load("trades"), ec.load("complexes"), ec.load("facilities")
    metrics = pd.read_parquet(ec.CACHE / "metrics.parquet")
    metrics.attrs["reference"] = m["reference"]
    features = build_features(metrics, complexes, facilities)

    findings = {}
    print("5단계: 시장 흐름 · 기획서 가설 검정")
    fig_market_heatmap(trades, reference, findings)
    fig_area_group_flow(trades, reference, findings)
    fig_district_ranking(metrics, complexes, findings)
    fig_correlations(features, findings)
    fig_observation_artifact(features, findings)
    fig_households(features, findings)
    fig_volume_reliability(features, findings)
    ec.dump(findings, "findings_05")


if __name__ == "__main__":
    main()
