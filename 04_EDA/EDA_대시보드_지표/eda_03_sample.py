"""3단계: 대시보드가 낙폭·회복률을 '띄울 수 있는가'를 결정하는 표본 조건을 본다.

metrics.py의 게이트는 두 개다.
  · 완결 분기 중 분기 거래 3건 이상인 관측이 4개 이상  (장기 이력)
  · 기준 분기 거래 3건 이상                              (현재가격)
둘 중 하나라도 못 넘으면 화면에는 '장기 자료 부족'만 뜬다. 이 게이트가 실제 데이터에서
어디를 얼마나 걸러내는지가 이 서비스의 실질 커버리지다.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import eda_common as ec

ec.setup()
STAGES = ["지표 전체 표시", "기준 분기 표본 부족", "장기 이력 부족", "이력·기준 분기 모두 부족"]
STAGE_COLORS = [ec.STATUS["good"], ec.STATUS["warning"], ec.STATUS["serious"], ec.STATUS["critical"]]


def fig_cell_counts(quarterly, reference, findings):
    """분기 셀의 거래 건수 분포. 임계값 3건이 어디를 자르는지 본다."""
    q = quarterly.loc[quarterly.quarter <= reference]
    counts = q["count"]
    capped = counts.clip(upper=20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    bins = np.arange(0.5, 21.5, 1)
    below = counts < ec.MIN_QUARTER_TRADES
    ax1.hist([capped[below], capped[~below]], bins=bins, stacked=True,
             color=[ec.STATUS["critical"], ec.CAT[0]], edgecolor=ec.SURFACE, linewidth=0.6,
             label=[f"3건 미만 — 지표 계산에서 제외 ({below.mean() * 100:.1f}%)",
                    f"3건 이상 — 자격 관측 ({(~below).mean() * 100:.1f}%)"])
    ax1.axvline(ec.MIN_QUARTER_TRADES - 0.5, color=ec.INK, linewidth=1.6)
    ax1.set_xlabel("단지×면적군×분기 셀의 거래 건수 (20건 이상은 20으로 묶음)")
    ax1.set_ylabel("셀 수")
    ax1.set_xticks([1, 3, 5, 10, 15, 20], ["1", "3", "5", "10", "15", "20+"])
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax1.legend(loc="upper right")
    ec.grid_y(ax1)
    ax1.set_title(f"분기 셀 거래 건수 (전체 {len(q):,}셀)", fontsize=10.5)

    # 셀은 걸러도 거래는 얼마나 남는가 — 셀 기준과 거래량 기준의 손실은 크게 다르다.
    kept_cells = (~below).mean() * 100
    kept_trades = counts[~below].sum() / counts.sum() * 100
    ax2.bar([0, 1], [100, 100], color=ec.GRID, width=0.5)
    ax2.bar([0, 1], [kept_cells, kept_trades], color=ec.CAT[0], width=0.5)
    ax2.set_xticks([0, 1], ["분기 셀 기준", "거래 건수 기준"])
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("3건 임계 통과 비율 (%)")
    for x, v in [(0, kept_cells), (1, kept_trades)]:
        ax2.text(x, v + 3, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color=ec.INK)
    ec.grid_y(ax2)
    ax2.set_title("임계값을 넘겨 남는 비중", fontsize=10.5)

    findings["cell_counts"] = {
        "전체 분기 셀": int(len(q)), "3건 미만 셀 비중": round(below.mean() * 100, 1),
        "셀 중위 거래건수": float(counts.median()),
        "3건 임계 통과 셀 비중": round(kept_cells, 1), "3건 임계 통과 거래 비중": round(kept_trades, 1)}
    fig.suptitle("분기 셀의 절반 이상이 3건 미만이지만, 거래량으로는 대부분이 살아남는다",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    return ec.save(fig, "06_분기셀_거래건수", "셀 = 단지 × 면적군 × 분기. 임계값 3건은 metrics.MIN_QUARTER_TRADES다.")


def fig_stage(elig, findings):
    """조합이 어느 게이트에서 탈락하는가 — 서비스 커버리지의 실체."""
    counts = elig["stage"].value_counts().reindex(STAGES).fillna(0).astype(int)
    trades = elig.groupby("stage")["trades"].sum().reindex(STAGES).fillna(0).astype(int)
    total, total_trades = counts.sum(), trades.sum()

    fig, ax = plt.subplots(figsize=(10.6, 3.5))
    for row, (series, denom, label) in enumerate([(counts, total, "단지×면적군 조합"),
                                                  (trades, total_trades, "거래 건수")]):
        left = 0
        for stage, color in zip(STAGES, STAGE_COLORS):
            width = series[stage] / denom * 100
            ax.barh(row, width, left=left, color=color, height=0.5)
            if width >= 6:
                ax.text(left + width / 2, row, f"{width:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=ec.SURFACE)
            # 2px 표면 간격으로 인접 채움을 분리한다.
            ax.barh(row, 0.35, left=left + width - 0.35 if left + width < 100 else 100,
                    color=ec.SURFACE, height=0.5)
            left += width
        ax.text(-1.5, row, label, ha="right", va="center", fontsize=10, color=ec.INK)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.6)
    ax.invert_yaxis()
    ax.set_xlabel("비중 (%)")
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS]
    ax.legend(handles, [f"{s} ({counts[s]:,}조합)" for s in STAGES], loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=2)
    full = counts["지표 전체 표시"]
    ax.set_title(f"낙폭·회복률이 실제로 표시되는 조합은 {full:,}개 / {total:,}개 ({full / total * 100:.1f}%)\n"
                 f"다만 그 조합이 거래량의 {trades['지표 전체 표시'] / total_trades * 100:.1f}%를 담고 있어, 사람들이 많이 찾는 단지는 대체로 보인다")
    findings["stage"] = {s: {"조합": int(counts[s]), "조합 비중": round(counts[s] / total * 100, 1),
                             "거래": int(trades[s]), "거래 비중": round(trades[s] / total_trades * 100, 1)}
                         for s in STAGES}
    return ec.save(fig, "07_지표표시_커버리지", "색은 상태값(양호→심각)이며 계열 색이 아니다.")


def fig_coverage_heatmap(elig, complexes, findings):
    """자치구 × 면적군으로 커버리지가 어떻게 갈리는지."""
    from housing.models import AREA_LABELS
    merged = elig.merge(complexes[["complex_id", "district"]], on="complex_id", how="left")
    pivot = merged.pivot_table(index="district", columns="area_group", values="full_metrics",
                               aggfunc="mean") * 100
    sizes = merged.pivot_table(index="district", columns="area_group", values="full_metrics", aggfunc="size")
    order = [a for a in AREA_LABELS if a in pivot.columns]
    pivot, sizes = pivot[order], sizes[order]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    sizes = sizes.loc[pivot.index]
    # 조합이 5개 미만인 칸은 비율이 의미 없으므로 비운다.
    masked = pivot.mask(sizes < 5)

    fig, ax = plt.subplots(figsize=(9.4, 8.4))
    sns.heatmap(masked, ax=ax, cmap=ec.SEQ, vmin=0, vmax=60, annot=True, fmt=".0f",
                annot_kws={"fontsize": 8.5}, linewidths=2, linecolor=ec.SURFACE,
                cbar_kws={"label": "지표 표시 가능률 (%)", "shrink": 0.45, "pad": 0.03})
    for (i, j), size in np.ndenumerate(sizes.values):
        if size < 5:
            ax.text(j + 0.5, i + 0.5, "표본\n부족", ha="center", va="center", fontsize=7, color=ec.MUTED)
    ax.set_xlabel("")
    ax.set_ylabel("")
    short = {"40㎡ 미만": "~40㎡", "135㎡ 이상": "135㎡~"}
    ax.set_xticklabels([short.get(t.get_text(), t.get_text().replace(" 이상 ", "~\n").replace(" 미만", ""))
                        for t in ax.get_xticklabels()], rotation=0, fontsize=8.5)
    ax.set_title("자치구 × 면적군별 지표 표시 가능률 (%)\n"
                 "고가 지역이 아니라 회전이 빠른 노원·도봉·성북의 중소형에서 지표가 가장 잘 뜬다")
    best = masked.stack().idxmax()
    findings["coverage_heatmap"] = {
        "최고": {"자치구": best[0], "면적군": best[1], "표시율": round(masked.stack().max(), 1)},
        "자치구 평균 상위": masked.mean(axis=1).nlargest(5).round(1).to_dict(),
        "자치구 평균 하위": masked.mean(axis=1).nsmallest(5).round(1).to_dict(),
        "면적군 평균": masked.mean().round(1).to_dict()}
    return ec.save(fig, "08_커버리지_히트맵", "연속형 값이라 파랑 단일 램프를 쓴다. 조합 5개 미만 칸은 비웠다.")


def fig_history_length(elig, findings):
    """관측 분기 수와 자격 분기 수의 간극 — '거래는 있는데 지표가 안 뜨는' 이유."""
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    bins = np.arange(0, 40, 1)
    ax.hist(elig["observed_quarters"].clip(upper=39), bins=bins, color=ec.CAT[0], alpha=0.85,
            edgecolor=ec.SURFACE, linewidth=0.5, label="거래가 한 건이라도 있던 분기 수")
    ax.hist(elig["qualified_quarters"].clip(upper=39), bins=bins, color=ec.CAT[1], alpha=0.85,
            edgecolor=ec.SURFACE, linewidth=0.5, label="3건 이상인 완결 분기 수 (자격 분기)")
    ax.axvline(ec.MIN_COMPLETE_QUARTERS, color=ec.INK, linewidth=1.6)
    ax.text(ec.MIN_COMPLETE_QUARTERS + 0.4, ax.get_ylim()[1] * 0.92,
            f"자격 분기 {ec.MIN_COMPLETE_QUARTERS}개 이상 필요", fontsize=9, color=ec.INK)
    ax.set_xlabel("분기 수 (조합당)")
    ax.set_ylabel("조합 수")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.legend(loc="upper right")
    ec.grid_y(ax)
    med_obs, med_qual = elig["observed_quarters"].median(), elig["qualified_quarters"].median()
    ax.set_title(f"관측 분기 중위 {med_obs:.0f}개 → 자격 분기 중위 {med_qual:.0f}개\n"
                 "거래 이력이 길어도 분기당 3건을 못 채우면 낙폭·회복률 계산에 들어가지 못한다")
    findings["history"] = {"관측 분기 중위": float(med_obs), "자격 분기 중위": float(med_qual),
                           "자격 분기 0개 조합 비중": round(elig["qualified_quarters"].eq(0).mean() * 100, 1)}
    return ec.save(fig, "09_이력_길이", "두 분포를 겹쳐 그렸고, 39분기 이상은 마지막 칸에 묶었다.")


def main():
    import sys
    sys.path.insert(0, str(ec.ROOT / "대시보드"))
    m = ec.meta()
    reference = pd.Period(m["reference"], freq="Q")
    quarterly, complexes = ec.load("quarterly"), ec.load("complexes")
    elig = ec.eligibility(quarterly, reference)
    elig.to_parquet(ec.CACHE / "eligibility.parquet", index=False)

    findings = {}
    print("3단계: 표본 충분성 · 지표 표시 커버리지")
    fig_cell_counts(quarterly, reference, findings)
    fig_stage(elig, findings)
    fig_coverage_heatmap(elig, complexes, findings)
    fig_history_length(elig, findings)
    ec.dump(findings, "findings_03")


if __name__ == "__main__":
    main()
