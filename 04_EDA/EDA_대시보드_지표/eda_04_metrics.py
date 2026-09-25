"""4단계: 대시보드 카드에 실제로 찍히는 지표(MDD·회복률·전고점 격차·회복 소요분기)의 분포.

지금까지는 '값이 뜨는가'를 봤고, 여기서는 '뜨는 값이 어떻게 생겼는가'를 본다.
기획서가 정한 회복률 해석 구간(50/80/100)이 이 데이터의 분포와 맞는지도 같이 확인한다.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import eda_common as ec

ec.setup()


def fig_mdd(metrics, findings):
    """최대 낙폭 분포. 기획서 예시(-30%)가 이 시장에서 어디쯤인지 본다."""
    mdd = metrics["mdd"]
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    sns.histplot(mdd, bins=60, color=ec.CAT[0], ax=ax, edgecolor=ec.SURFACE, linewidth=0.4)
    median = mdd.median()
    ax.axvline(median, color=ec.INK, linewidth=1.8)
    ax.axvline(-30, color=ec.STATUS["critical"], linestyle=(0, (4, 3)), linewidth=1.6)
    top = ax.get_ylim()[1]
    ax.text(median, top * 0.97, f" 중위 {median:.1f}%", fontsize=9.5, color=ec.INK, va="top", fontweight="bold")
    ax.text(-30, top * 0.80, "기획서 예시 -30% ", fontsize=9, color=ec.STATUS["critical"], va="top", ha="right")
    ax.set_xlabel("최대 낙폭 MDD (%)   ·   전고점 대비 저점 하락률")
    ax.set_ylabel("단지×면적군 조합 수")
    ec.grid_y(ax)
    share30 = (mdd <= -30).mean() * 100
    findings["mdd"] = {"조합 수": int(len(mdd)), "중위": round(median, 2), "평균": round(mdd.mean(), 2),
                       "p10": round(mdd.quantile(0.10), 2), "p90": round(mdd.quantile(0.90), 2),
                       "-30% 이하 비중": round(share30, 1), "최저": round(mdd.min(), 2),
                       "하락 없음 조합": int(metrics["no_decline"].sum())}
    ax.set_title(f"최대 낙폭 분포 (지표 산출 {len(mdd):,}개 조합)\n"
                 f"중위 {median:.1f}% — 기획서 예시로 든 -30%보다 깊은 조합이 {share30:.1f}%나 된다")
    return ec.save(fig, "10_MDD_분포", "MDD는 metrics.py와 같이 음수로 관리한다. 값이 작을수록 깊은 하락이다.")


def fig_recovery(metrics, findings):
    """회복률 분포와 기획서 해석 구간의 적합성."""
    rec = metrics["recovery"].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    # 양끝 막대는 잘린 꼬리 전부를 담으므로, 값의 밀도가 아니라 '그 바깥의 총량'이다. 따로 표시한다.
    low, high = -50, 250
    inner = rec[(rec >= low) & (rec <= high)]
    over, under = int((rec > high).sum()), int((rec < low).sum())
    sns.histplot(inner, bins=60, color=ec.CAT[0], ax=ax1, edgecolor=ec.SURFACE, linewidth=0.4)
    top = max(ax1.get_ylim()[1], over * 1.25)
    ax1.bar([high + 4], [over], width=8, color=ec.CAT[1], zorder=3)
    ax1.annotate(f"{high}% 초과\n{over:,}개 조합", (high + 4, over), textcoords="offset points",
                 xytext=(-4, 6), ha="right", fontsize=8.5, color=ec.CAT[1], fontweight="bold")
    for edge in [50, 80, 100]:
        ax1.axvline(edge, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    for x, label in [(5, "회복\n지연"), (65, "회복\n진행"), (90, "전고점\n근접"), (165, "전고점 회복·돌파")]:
        ax1.text(x, top * 0.99, label, fontsize=8.5, color=ec.INK2, ha="center", va="top")
    ax1.axvline(rec.median(), color=ec.INK, linewidth=1.8)
    ax1.text(rec.median(), top * 0.60, f" 중위 {rec.median():.0f}%", fontsize=9.5,
             color=ec.INK, fontweight="bold")
    ax1.set_ylim(0, top * 1.02)
    if under:
        ax1.text(low, top * 0.5, f" {low}% 미만 {under:,}개", fontsize=8.5, color=ec.INK2)
    ax1.set_xlabel("하락분 회복률 (%)   ·   (현재가 - 저점) ÷ (전고점 - 저점)")
    ax1.set_ylabel("조합 수")
    ec.grid_y(ax1)
    ax1.set_title("회복률 분포와 기획서 해석 구간", fontsize=10.5)

    bands = [b[2] for b in ec.RECOVERY_BANDS]
    counts = metrics["band"].value_counts().reindex(bands).fillna(0).astype(int)
    colors = [b[3] for b in ec.RECOVERY_BANDS]
    ax2.barh(np.arange(len(bands)), counts.values, color=colors, height=0.62)
    ax2.set_yticks(np.arange(len(bands)), bands)
    ax2.invert_yaxis()
    ax2.set_xlabel("조합 수")
    ec.grid_x(ax2)
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax2.text(v + total * 0.012, i, f"{v:,}  ({v / total * 100:.1f}%)", va="center",
                 fontsize=9, color=ec.INK2)
    ax2.set_xlim(0, counts.max() * 1.32)
    ax2.set_title("구간별 조합 수", fontsize=10.5)

    findings["recovery"] = {
        "조합 수": int(len(rec)), "중위": round(rec.median(), 1), "평균": round(rec.mean(), 1),
        "p10": round(rec.quantile(0.10), 1), "p90": round(rec.quantile(0.90), 1),
        "100% 이상 비중": round((rec >= 100).mean() * 100, 1),
        "50% 미만 비중": round((rec < 50).mean() * 100, 1),
        "음수 비중": round((rec < 0).mean() * 100, 1),
        "구간별": {b: int(counts[b]) for b in bands}}
    fig.suptitle(f"회복률 중위 {rec.median():.0f}% — 조합의 {(rec >= 100).mean() * 100:.1f}%가 이미 전고점을 회복·돌파했다. "
                 f"기획서의 4구간 중 상단 하나에 표본이 쏠린다",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    return ec.save(fig, "11_회복률_분포", "가로축은 -50~250%로 잘랐고 초과분은 주황 막대로 따로 표시했다. 통계값과 구간 집계는 전체 표본 기준이다.")


def fig_recovery_instability(metrics, complexes, findings):
    """회복률의 분모는 (전고점 - 저점)이다. 낙폭이 얕으면 분모가 0에 수렴해 값이 폭발한다.

    이건 계산 오류가 아니라 정의에서 오는 성질이지만, 대시보드는 이 값을 그대로 비교표에 찍는다.
    얕은 낙폭 구간을 얼마나 잘라야 지표가 읽을 만해지는지를 본다.
    """
    d = metrics.dropna(subset=["recovery"]).copy()
    d["abs_mdd"] = d["mdd"].abs()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})

    # 로그 축이므로 0 이하 회복률(현재가가 저점보다 낮은 조합)은 빼고 각주로 건수를 남긴다.
    positive = d.loc[d["recovery"] > 0]
    ax1.scatter(positive["abs_mdd"], positive["recovery"], s=14, color=ec.CAT[0],
                alpha=0.45, linewidths=0)
    ax1.set_yscale("log")
    ax1.axhline(100, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax1.axvline(5, color=ec.STATUS["critical"], linewidth=1.6)
    ax1.text(5.4, 6e4, "|MDD| 5% 미만\n분모가 얇아 값이 폭발하는 구간",
             fontsize=9, color=ec.STATUS["critical"], va="top")
    ax1.text(0.985, 0.04,
             f"가로선 = 전고점 회복(100%)  ·  회복률 0 이하 {len(d) - len(positive)}개 조합은 로그 축에서 제외",
             transform=ax1.transAxes, ha="right", fontsize=8.5, color=ec.INK2)
    ax1.set_xlabel("최대 낙폭의 크기 |MDD| (%)")
    ax1.set_ylabel("하락분 회복률 (%, 로그)")
    ec.grid_y(ax1)
    worst = d.loc[d["recovery"].idxmax()]
    name = complexes.set_index("complex_id").loc[worst.complex_id, "name"]
    ax1.annotate(f"최댓값 {worst.recovery:,.0f}%\n{name} · |MDD| {worst.abs_mdd:.2f}%",
                 (worst.abs_mdd, worst.recovery), textcoords="offset points", xytext=(52, -34),
                 fontsize=8.5, color=ec.INK, arrowprops={"arrowstyle": "-", "color": ec.MUTED, "linewidth": 0.8})
    ax1.set_title("낙폭이 얕을수록 회복률이 발산한다", fontsize=10.5)

    cuts = [0, 1, 2, 3, 5, 10]
    stats = [(c, d.loc[d.abs_mdd >= c, "recovery"]) for c in cuts]
    x = np.arange(len(cuts))
    ax2.plot(x, [s.median() for _, s in stats], marker="o", color=ec.CAT[0], label="중위")
    ax2.plot(x, [s.quantile(0.90) for _, s in stats], marker="s", color=ec.CAT[1], label="상위 10%")
    ax2.set_yscale("log")
    ax2.set_xticks(x, [f"전체" if c == 0 else f"≥{c}%" for c in cuts])
    ax2.set_xlabel("표본에 남기는 |MDD| 하한")
    ax2.set_ylabel("회복률 (%, 로그)")
    ax2.legend(loc="upper right")
    ec.grid_y(ax2)
    for xi, (_, s) in enumerate(stats):
        ax2.annotate(f"{s.median():,.0f}", (xi, s.median()), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=8, color=ec.INK2)
    ax2.set_title("중위값은 흔들리지 않지만 상위 10%가 극단으로 끌려간다", fontsize=10.5)

    shallow = d.abs_mdd.lt(5)
    findings["recovery_instability"] = {
        "회복률 최댓값": round(d.recovery.max(), 1),
        "최댓값 단지": str(name), "최댓값 |MDD|": round(worst.abs_mdd, 3),
        "|MDD|<5% 조합": int(shallow.sum()), "|MDD|<5% 비중": round(shallow.mean() * 100, 1),
        "|MDD|<5% 회복률 중위": round(d.loc[shallow, "recovery"].median(), 1),
        "회복률 1000% 초과 조합": int(d.recovery.gt(1000).sum()),
        "그중 |MDD|<5% 비중": round(d.loc[d.recovery.gt(1000), "abs_mdd"].lt(5).mean() * 100, 1),
        "|MDD|>=5% 회복률 중위": round(d.loc[~shallow, "recovery"].median(), 1),
        "|MDD|>=5% 회복률 평균": round(d.loc[~shallow, "recovery"].mean(), 1)}
    fig.suptitle(f"회복률 최댓값 {d.recovery.max():,.0f}% — 대시보드 비교표에 이 값이 그대로 찍힌다. "
                 f"원인은 얕은 낙폭({shallow.mean() * 100:.1f}%의 조합)이다",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    return ec.save(fig, "12_회복률_불안정구간",
                   "회복률 = (현재가 - 저점) ÷ (전고점 - 저점). 분모가 0에 가까우면 값이 발산한다.")


def fig_mdd_vs_recovery(metrics, findings):
    """기획서 핵심 가설: 낙폭이 작고 회복률이 높은 단지가 방어력이 좋다 — 두 축이 실제로 어떻게 놓이는가."""
    d = metrics.dropna(subset=["recovery"]).copy()
    d["rec_c"] = d["recovery"].clip(-50, 250)

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    ax.axhspan(100, 260, color=ec.STATUS["good"], alpha=0.05, zorder=0)
    ax.axhline(100, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.axvline(-30, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    sc = ax.scatter(d["mdd"], d["rec_c"], c=d["total_trades"], cmap=ec.SEQ, s=16,
                    linewidths=0.4, edgecolors=ec.SURFACE, norm="log", zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("자격 분기 누적 거래 건수 (로그)", fontsize=9)

    rho = d[["mdd", "recovery"]].corr(method="spearman").iloc[0, 1]
    # 얕은 낙폭 조합은 회복률이 발산하므로, 관계의 실체는 |MDD|>=5% 부분집합에서 읽는다.
    solid = d.loc[d["mdd"].abs() >= 5]
    rho_solid = solid[["mdd", "recovery"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("최대 낙폭 MDD (%)")
    ax.set_ylabel("하락분 회복률 (%)")
    ec.grid_y(ax)
    ec.grid_x(ax)
    ax.text(0.985, 0.03, f"스피어만 ρ = {rho:+.3f}   (|MDD|≥5% 부분집합 {rho_solid:+.3f})",
            transform=ax.transAxes, ha="right", fontsize=10, color=ec.INK, fontweight="bold")
    ax.text(-2, 235, "낙폭 작고 회복 완료\n(기획서가 말한 '방어력 높은 단지')", fontsize=8.5,
            color=ec.INK2, ha="right", va="top")
    deep_recovered = ((d.mdd <= -30) & (d.recovery >= 100)).mean() * 100
    ax.set_title(f"최대 낙폭 vs 회복률 — 얕게 빠진 조합일수록 회복률이 높다 (ρ = {rho:+.3f})\n"
                 f"단 이 관계의 상당 부분은 분모 효과다. 30% 넘게 빠지고도 전고점을 되찾은 조합도 {deep_recovered:.1f}% 있다")
    findings["mdd_vs_recovery"] = {
        "스피어만 ρ": round(rho, 3), "스피어만 ρ (|MDD|≥5%)": round(rho_solid, 3),
        "피어슨 r": round(d[["mdd", "recovery"]].corr().iloc[0, 1], 3),
        "낙폭 30%+ 이면서 회복 100%+ 비중": round(deep_recovered, 1),
        "낙폭 30% 미만이면서 회복 100% 미만 비중": round(((d.mdd > -30) & (d.recovery < 100)).mean() * 100, 1)}
    return ec.save(fig, "13_MDD_회복률_산점도",
                   "세로축은 250%에서 잘랐다. 점 색은 연속형 거래량이라 파랑 단일 램프를 쓴다.")


def fig_gap_and_speed(metrics, findings):
    """전고점 대비 현재 격차와, 저점 이후 회복까지 걸린 분기 수."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))

    gap = metrics["gap"].dropna()
    sns.histplot(gap.clip(-60, 120), bins=60, color=ec.CAT[0], ax=ax1,
                 edgecolor=ec.SURFACE, linewidth=0.4)
    ax1.axvline(0, color=ec.INK, linewidth=1.8)
    ax1.text(0, ax1.get_ylim()[1] * 0.97, " 전고점 = 0%", fontsize=9, color=ec.INK, va="top")
    ax1.set_xlabel("전고점 대비 현재 가격 차이 (%)")
    ax1.set_ylabel("조합 수")
    ec.grid_y(ax1)
    ax1.set_title(f"전고점 격차  ·  전고점 이상 {(gap >= 0).mean() * 100:.1f}%", fontsize=10.5)

    speed = metrics["recovery_quarters"].dropna()
    unrecovered = metrics["recovered"].eq(False).sum()
    bins = np.arange(0.5, speed.max() + 1.5, 1)
    ax2.hist(speed, bins=bins, color=ec.CAT[1], edgecolor=ec.SURFACE, linewidth=0.5)
    ax2.axvline(speed.median(), color=ec.INK, linewidth=1.8)
    ax2.text(speed.median(), ax2.get_ylim()[1] * 0.97, f" 중위 {speed.median():.0f}분기",
             fontsize=9.5, color=ec.INK, va="top", fontweight="bold")
    ax2.set_xlabel("저점 이후 전고점 회복까지 걸린 분기 수")
    ax2.set_ylabel("조합 수")
    ec.grid_y(ax2)
    ax2.set_title(f"회복 소요 기간  ·  아직 미회복 {unrecovered:,}개 조합", fontsize=10.5)

    findings["gap"] = {"중위": round(gap.median(), 1), "전고점 이상 비중": round((gap >= 0).mean() * 100, 1),
                       "p10": round(gap.quantile(0.10), 1), "p90": round(gap.quantile(0.90), 1)}
    findings["recovery_speed"] = {
        "회복 완료 조합": int(len(speed)), "미회복 조합": int(unrecovered),
        "미회복 비중": round(unrecovered / metrics["recovery"].notna().sum() * 100, 1),
        "중위 분기": float(speed.median()), "p90 분기": float(speed.quantile(0.90)), "최장": int(speed.max())}
    fig.suptitle(f"전고점을 되찾은 조합의 중위 소요 기간은 {speed.median():.0f}분기 "
                 f"— 나머지 {unrecovered / metrics['recovery'].notna().sum() * 100:.1f}%는 아직 회복 전이다",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    return ec.save(fig, "14_전고점격차_회복기간", "격차는 -60~120%로 잘라 그렸고, 통계값은 전체 표본 기준이다.")


def fig_peak_trough_timing(metrics, findings):
    """전고점·저점이 언제였나 — 지표가 같은 시장 충격을 보고 있는지 확인한다."""
    peaks = pd.PeriodIndex(metrics["peak_quarter"].dropna(), freq="Q").value_counts().sort_index()
    troughs = pd.PeriodIndex(metrics["trough_quarter"].dropna(), freq="Q").value_counts().sort_index()
    index = pd.period_range(min(peaks.index.min(), troughs.index.min()),
                            max(peaks.index.max(), troughs.index.max()), freq="Q")
    peaks, troughs = peaks.reindex(index, fill_value=0), troughs.reindex(index, fill_value=0)

    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    x = np.arange(len(index))
    ax.bar(x - 0.21, peaks.values, width=0.4, color=ec.CAT[0], label="전고점이 있던 분기")
    ax.bar(x + 0.21, troughs.values, width=0.4, color=ec.CAT[1], label="저점이 있던 분기")
    step = 2
    ax.set_xticks(x[::step], [f"{p.year}\nQ{p.quarter}" for p in index[::step]], fontsize=8)
    ax.set_ylabel("조합 수")
    ax.legend(loc="upper left")
    ec.grid_y(ax)
    top_peak, top_trough = peaks.idxmax(), troughs.idxmax()
    ax.annotate(f"전고점 최빈 {top_peak}", (list(index).index(top_peak) - 0.21, peaks.max()),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9,
                color=ec.CAT[0], fontweight="bold")
    ax.annotate(f"저점 최빈 {top_trough}", (list(index).index(top_trough) + 0.21, troughs.max()),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9,
                color=ec.CAT[1], fontweight="bold")
    ax.set_ylim(0, max(peaks.max(), troughs.max()) * 1.18)
    ax.set_title(f"전고점·저점 발생 분기 분포\n"
                 f"전고점은 {top_peak}, 저점은 {top_trough}에 몰린다 — 개별 단지가 아니라 같은 시장 충격을 보고 있다")
    findings["timing"] = {"전고점 최빈 분기": str(top_peak), "전고점 최빈 비중": round(peaks.max() / peaks.sum() * 100, 1),
                          "저점 최빈 분기": str(top_trough), "저점 최빈 비중": round(troughs.max() / troughs.sum() * 100, 1),
                          "전고점 2021년 비중": round(peaks[[p for p in index if p.year == 2021]].sum() / peaks.sum() * 100, 1),
                          "저점 2022~2023년 비중": round(troughs[[p for p in index if p.year in (2022, 2023)]].sum() / troughs.sum() * 100, 1)}
    return ec.save(fig, "15_전고점_저점_시점", "두 계열이므로 범례를 두고 최빈 분기는 직접 라벨했다.")


def main():
    m = ec.meta()
    reference = pd.Period(m["reference"], freq="Q")
    quarterly, complexes = ec.load("quarterly"), ec.load("complexes")
    metrics = ec.compute_metrics(quarterly, reference)
    metrics.to_parquet(ec.CACHE / "metrics.parquet", index=False)

    findings = {}
    print(f"4단계: 핵심 지표 분포 (지표 산출 조합 {len(metrics):,}개)")
    fig_mdd(metrics, findings)
    fig_recovery(metrics, findings)
    fig_recovery_instability(metrics, complexes, findings)
    fig_mdd_vs_recovery(metrics, findings)
    fig_gap_and_speed(metrics, findings)
    fig_peak_trough_timing(metrics, findings)
    ec.dump(findings, "findings_04")


if __name__ == "__main__":
    main()
