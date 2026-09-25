"""2단계: 대시보드에 값이 뜨기 전, 입력 데이터의 품질과 원자료 분포를 본다.

대시보드가 보여주는 모든 숫자는 (1) 유효 거래, (2) ㎡당 단가, (3) 단지 속성,
(4) 최근접 시설 거리에서 나온다. 이 네 재료의 결측과 분포를 먼저 확인한다.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import eda_common as ec

ec.setup()
FACILITY_ORDER = ["지하철", "학교", "공원", "대규모점포"]


def fig_completeness(trades, complexes, facilities, findings):
    """대시보드 카드·비교표·시설 목록이 의존하는 필드의 채움률."""
    n_c = len(complexes)
    items = [
        ("거래 식별 (단지명·시군구)", trades["단지명"].ne("").mean() * 100, len(trades), "거래"),
        ("KAPT 단지코드 매칭", trades["matched"].mean() * 100, len(trades), "거래"),
        ("단지 좌표 (위도·경도)", complexes["latitude"].notna().mean() * 100, n_c, "단지"),
        ("건축년도 (연식)", complexes["built_year"].notna().mean() * 100, n_c, "단지"),
        ("세대수", complexes["households"].notna().mean() * 100, n_c, "단지"),
        ("주차 대수", complexes[["parking_ground", "parking_underground"]].notna().any(axis=1).mean() * 100, n_c, "단지"),
    ]
    for kind in FACILITY_ORDER:
        sub = facilities.loc[facilities.kind == kind]
        items.append((f"{kind} 1km 내 유효 매핑", sub["status"].eq("ok").mean() * 100, len(sub), "단지"))

    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    colors = [ec.STATUS["good"] if v >= 95 else ec.STATUS["warning"] if v >= 70
              else ec.STATUS["serious"] if v >= 40 else ec.STATUS["critical"] for v in values]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    y = np.arange(len(items))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("채움률 (%)")
    ec.grid_x(ax)
    for yi, (v, item) in enumerate(zip(values, items)):
        ax.text(v + 1.5, yi, f"{v:.1f}%  ({item[2]:,}{item[3]} 중)", va="center", fontsize=8.5, color=ec.INK2)
    ax.set_title("대시보드가 쓰는 필드별 채움률\n세대수·주차·공원 거리는 결측이 커서 비교 지표로 쓰기 전 확인이 필요하다")
    findings["completeness"] = {label: round(v, 1) for label, v in zip(labels, values)}
    return ec.save(fig, "01_필드_채움률", "색은 상태값(95%↑ 양호 / 70%↑ 주의 / 40%↑ 경고 / 그 미만 심각)이며 계열 색이 아니다.")


def fig_volume(trades, meta, findings):
    """거래량 시계열: 기획서가 말한 고점·조정·회복 구간이 실제로 표본에 들어있는지 본다."""
    monthly = trades.groupby(trades["date"].dt.to_period("M")).size()
    monthly.index = monthly.index.to_timestamp()
    ref_end = pd.Period(meta["reference"], freq="Q").end_time

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.fill_between(monthly.index, monthly.values, color=ec.CAT[0], alpha=0.16, zorder=2)
    ax.plot(monthly.index, monthly.values, color=ec.CAT[0], zorder=3)
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"), color=ec.GRID, alpha=0.55, zorder=1)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"), color=ec.GRID, alpha=0.9, zorder=1)
    ax.axvline(ref_end, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.4, zorder=4)
    top = monthly.max()
    for x, label in [(pd.Timestamp("2021-07-01"), "가격 고점기"), (pd.Timestamp("2023-01-01"), "가격 조정기"),
                     (pd.Timestamp("2025-02-01"), "가격 회복기")]:
        ax.text(x, top * 0.94, label, fontsize=9, color=ec.INK2, ha="center")
    ax.text(ref_end, top * 0.55, f" 기준 분기\n {meta['reference']}", fontsize=8.5, color=ec.INK2, va="center")
    ax.set_ylabel("월 거래 건수")
    ax.set_ylim(0, top * 1.05)
    ec.grid_y(ax)
    ax.set_title(f"월별 유효 거래량 ({meta['start']} ~ {meta['end']})\n"
                 f"기획서가 요구한 고점·조정·회복 세 구간이 모두 표본에 들어 있다")
    trough = monthly.idxmin()
    findings["volume"] = {"월 최다": {"시점": str(monthly.idxmax().date()), "건수": int(monthly.max())},
                          "월 최소": {"시점": str(trough.date()), "건수": int(monthly.min())},
                          "2022~2023 평균": round(monthly["2022":"2023"].mean(), 1),
                          "2021 평균": round(monthly["2021"].mean(), 1),
                          "2025년 이후 평균": round(monthly["2025":].mean(), 1)}
    return ec.save(fig, "02_월별_거래량", "세로축은 거래량이고, 회색 띠와 라벨은 기획서가 정의한 가격 사이클 구간이다.")


def fig_price_distributions(trades, findings):
    """가격·면적·㎡당 단가: 대시보드 대표값이 '㎡당 중위가격'인 이유를 확인한다."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    specs = [("amount", "거래금액 (만원)", ec.CAT[0]), ("area", "전용면적 (㎡)", ec.CAT[1]),
             ("unit_price", "㎡당 단가 (만원/㎡)", ec.CAT[2])]
    for ax, (col, label, color) in zip(axes, specs):
        values = trades[col]
        upper = values.quantile(0.995)
        sns.histplot(values[values <= upper], bins=70, color=color, ax=ax,
                     edgecolor=ec.SURFACE, linewidth=0.4)
        mean, median = values.mean(), values.median()
        ax.axvline(median, color=ec.INK, linewidth=1.6)
        ax.axvline(mean, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.4)
        ax.text(median, ax.get_ylim()[1] * 0.97, f" 중위 {median:,.0f}", fontsize=8.5, color=ec.INK, va="top")
        ax.text(mean, ax.get_ylim()[1] * 0.86, f" 평균 {mean:,.0f}", fontsize=8.5, color=ec.INK2, va="top")
        ax.set_xlabel(label)
        ax.set_ylabel("거래 건수" if col == "amount" else "")
        ec.grid_y(ax)
        skew = values.skew()
        findings.setdefault("distributions", {})[col] = {
            "평균": round(mean, 1), "중위": round(median, 1), "왜도": round(skew, 2),
            "p99": round(values.quantile(0.99), 1), "최대": round(values.max(), 1)}
        ax.set_title(f"{label.split(' (')[0]}  ·  왜도 {skew:+.2f}", fontsize=10.5)
    fig.suptitle("세 원자료 모두 오른쪽으로 길게 치우쳐 평균이 중위값을 웃돈다 — 대표값으로 중위값이 맞다",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.04)
    fig.tight_layout()
    return ec.save(fig, "03_원자료_분포", "가로축은 상위 0.5% 이상치를 잘라 표시했고, 통계값은 전체 표본으로 계산했다.")


def fig_area_groups(trades, findings):
    """면적군은 대시보드의 필수 선택값이다. 군별 표본 크기와 단가 수준을 같이 본다."""
    from housing.models import AREA_LABELS  # noqa: E402
    order = [a for a in AREA_LABELS if a in set(trades.area_group)]
    counts = trades.area_group.value_counts().reindex(order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    ax1.barh(np.arange(len(order)), counts.values, color=ec.CAT[0], height=0.62)
    ax1.set_yticks(np.arange(len(order)), order)
    ax1.invert_yaxis()
    ax1.set_xlabel("거래 건수")
    ax1.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ec.grid_x(ax1)
    for i, v in enumerate(counts.values):
        ax1.text(v + counts.max() * 0.015, i, f"{v:,}  ({v / counts.sum() * 100:.1f}%)",
                 va="center", fontsize=8.5, color=ec.INK2)
    ax1.set_title("면적군별 표본 크기", fontsize=10.5)

    sub = trades.loc[trades.unit_price <= trades.unit_price.quantile(0.995)]
    sns.boxplot(data=sub, y="area_group", x="unit_price", order=order, ax=ax2, orient="h",
                color=ec.CAT[0], width=0.6, fliersize=0,
                boxprops={"edgecolor": ec.INK2, "linewidth": 0.8, "alpha": 0.85},
                medianprops={"color": ec.SURFACE, "linewidth": 2},
                whiskerprops={"color": ec.INK2, "linewidth": 0.8},
                capprops={"color": ec.INK2, "linewidth": 0.8})
    ax2.set_yticks(np.arange(len(order)), ["" for _ in order])
    ax2.set_ylabel("")
    ax2.set_xlabel("㎡당 단가 (만원/㎡)")
    ec.grid_x(ax2)
    medians = trades.groupby("area_group", observed=True)["unit_price"].median().reindex(order)
    for i, v in enumerate(medians.values):
        ax2.text(v, i - 0.36, f"중위 {v:,.0f}", fontsize=8.5, color=ec.INK, ha="center")
    ax2.set_title("면적군별 ㎡당 단가 분포", fontsize=10.5)

    findings["area_groups"] = {a: {"거래": int(counts[a]), "비중": round(counts[a] / counts.sum() * 100, 1),
                                  "중위단가": round(medians[a], 1)} for a in order}
    fig.suptitle("‘85㎡ 이상 102㎡ 미만’은 표본이 가장 얇아 이 면적군의 지표는 흔들리기 쉽다",
                 fontsize=12, fontweight="bold", x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    return ec.save(fig, "04_면적군_표본과_단가", "박스플롯은 상위 0.5% 이상치를 제외해 그렸다.")


def fig_facility_distance(facilities, findings):
    """시설 거리: 1km를 넘으면 대시보드가 'invalid'로 처리해 화면에서 빠진다."""
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    valid = facilities.loc[facilities.distance.notna()]
    for i, kind in enumerate(FACILITY_ORDER):
        d = valid.loc[valid.kind == kind, "distance"]
        sns.kdeplot(d.clip(upper=2500), ax=ax, color=ec.CAT[i], linewidth=2,
                    label=f"{kind} (중위 {d.median():,.0f}m)", clip=(0, 2500))
    ax.axvline(1000, color=ec.INK, linestyle=(0, (4, 3)), linewidth=1.5)
    ax.text(1015, ax.get_ylim()[1] * 0.95, "1km 유효 경계\n초과분은 화면에서 제외", fontsize=8.5,
            color=ec.INK, va="top")
    ax.set_xlabel("최근접 시설까지 직선거리 (m)")
    ax.set_ylabel("밀도")
    ax.set_xlim(0, 2500)
    ax.legend(loc="upper right")
    ec.grid_y(ax)
    over = valid.assign(over=valid.distance.gt(1000)).groupby("kind")["over"].mean() * 100
    findings["facility_over_1km"] = {k: round(over[k], 1) for k in FACILITY_ORDER}
    ax.set_title("공원만 절반 가까이가 1km 밖 — 접근성 가설 검정에서 공원은 표본이 반토막 난다")
    return ec.save(fig, "05_시설거리_분포", "가로축은 2.5km에서 잘랐다. 중위값은 전체 표본 기준이다.")


def main():
    import sys
    sys.path.insert(0, str(ec.ROOT / "대시보드"))
    trades, complexes, facilities = ec.load("trades"), ec.load("complexes"), ec.load("facilities")
    m = ec.meta()
    findings = {"meta": {k: m[k] for k in ["csv_mb", "start", "end", "reference", "partial"]},
                "diagnostics": m["diagnostics"]}
    print("2단계: 데이터 품질 · 원자료 분포")
    fig_completeness(trades, complexes, facilities, findings)
    fig_volume(trades, m, findings)
    fig_price_distributions(trades, findings)
    fig_area_groups(trades, findings)
    fig_facility_distance(facilities, findings)
    ec.dump(findings, "findings_02")


if __name__ == "__main__":
    main()
