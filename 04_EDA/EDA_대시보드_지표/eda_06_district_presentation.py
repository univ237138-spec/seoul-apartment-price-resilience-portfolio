"""발표용: 자치구 25개의 MDD × 회복 위치를 산점도와 롤리팝 두 방식으로 시각화한다.

지난 EDA(그림 18)는 2026Q2 기준이었다. 이 스크립트는 대시보드가 지금 실제로 쓰는 파이프라인
(기준 분기 완화 포함, 2026Q3 기준)으로 다시 집계해 발표 시점 수치와 어긋나지 않게 한다.
회복률 대신 '고점 대비 위치'를 쓰는 이유, |MDD|>=5%만 쓰는 이유는 eda_04_metrics.py의
회복률 불안정구간 분석과 동일하다 — 분모(고점-저점)가 얕은 낙폭에서 0에 수렴해 값이 발산하기 때문.
"""
import sys
sys.path.insert(0, '../대시보드')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_common as ec
from housing.data import load_csv, DEFAULT_CSV
from housing.scoring import axis_inputs

ec.setup()
SOLID_MDD = 5.0
MIN_COMBOS = 15


def load_district_table():
    data = load_csv(DEFAULT_CSV)
    table = axis_inputs(data.quarterly, data.reference, data.partial)
    table = table.merge(data.complexes[['complex_id', 'district']], on='complex_id', how='left')
    solid = table.loc[table.mdd.abs() >= SOLID_MDD].dropna(subset=['gap'])
    g = solid.groupby('district').agg(mdd=('mdd', 'median'), gap=('gap', 'median'), n=('mdd', 'size'))
    g = g.loc[g.n >= MIN_COMBOS].sort_values('mdd')
    return g, str(data.reference)


def fig_scatter(g, reference):
    """25개 자치구를 MDD × 고점 대비 위치로 흩뿌려 관계 자체를 보여준다."""
    fig, ax = plt.subplots(figsize=(11, 9))
    med_mdd, med_gap = g.mdd.median(), g.gap.median()

    # 사분면 배경 — 오른쪽 위(얕게 빠지고 회복)만 옅게 강조
    ax.axhspan(med_gap, g.gap.max() + 8, xmin=0, xmax=1, color=ec.STATUS['good'], alpha=0.05, zorder=0)
    ax.axvline(med_mdd, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.axhline(med_gap, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.axhline(0, color=ec.INK, linewidth=1, zorder=1)

    sizes = 90 + (g.n / g.n.max()) * 260
    ax.scatter(g.mdd, g.gap, s=sizes, color=ec.CAT[0], alpha=0.82,
              edgecolors='white', linewidths=1.4, zorder=3)

    # 좌하단(금천·노원·도봉)이 뭉쳐 있어 축 여백을 넉넉히 두고, 그 클러스터의 라벨만
    # 점 아래쪽으로 내려 사분면 안내문과 겹치지 않게 한다.
    xpad = (g.mdd.max() - g.mdd.min()) * 0.22
    ypad = (g.gap.max() - g.gap.min()) * 0.16
    xlim = (g.mdd.min() - xpad, g.mdd.max() + xpad * 0.5)
    ylim = (g.gap.min() - ypad, g.gap.max() + ypad)
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    crowded = {'금천구', '노원구', '도봉구'}
    for name, row in g.iterrows():
        if name in crowded:
            xytext, va = (0, -13), 'top'
        else:
            xytext = (6, 4) if row.mdd >= med_mdd else (-6, 4)
        ha = 'center' if name in crowded else ('left' if xytext[0] > 0 else 'right')
        va = 'top' if name in crowded else 'bottom'
        ax.annotate(name, (row.mdd, row.gap), textcoords='offset points', xytext=xytext,
                   fontsize=9, color=ec.INK, ha=ha, va=va, fontweight='bold' if name in
                   ['강남구', '송파구', '노원구', '도봉구', '용산구', '금천구'] else 'normal')

    pad_x, pad_y = (xlim[1] - xlim[0]) * 0.02, (ylim[1] - ylim[0]) * 0.03
    ax.text(xlim[0] + pad_x, ylim[1] - pad_y, '깊게 빠졌지만\n회복', fontsize=9.5,
           color=ec.INK2, ha='left', va='top')
    ax.text(xlim[1] - pad_x, ylim[1] - pad_y, '얕게 빠지고\n회복', fontsize=9.5,
           color=ec.STATUS['good'], ha='right', va='top', fontweight='bold')
    ax.text(xlim[1] - pad_x, ylim[0] + pad_y, '얕게 빠졌지만\n미회복', fontsize=9.5,
           color=ec.INK2, ha='right', va='bottom')
    # 이 사분면은 금천·노원·도봉 클러스터가 맨 아래를 차지하고 있어, 안내문은
    # 그 클러스터 라벨보다 위쪽(축 하단에서 22% 지점)에 별도로 띄운다.
    ax.text(xlim[0] + pad_x, ylim[0] + (ylim[1] - ylim[0]) * 0.22, '깊게 빠지고\n미회복', fontsize=9.5,
           color=ec.STATUS['critical'], ha='left', va='bottom', fontweight='bold')

    ax.set_xlabel('최대 낙폭 MDD 중위값 (%) · 오른쪽일수록 얕게 하락')
    ax.set_ylabel('전고점 대비 위치 중위값 (%) · 0%는 전고점 회복, 위쪽일수록 더 높음')
    ec.grid_x(ax); ec.grid_y(ax)
    ax.set_title(f'자치구 25곳 · 최대 낙폭 × 전고점 대비 위치 (기준 {reference})\n'
                f'점 크기 = 표본 조합 수 · 점선은 25개 자치구 중위값(낙폭 {med_mdd:.1f}% · 위치 {med_gap:+.1f}%)')
    return ec.save(fig, '23_자치구_산점도_MDD_고점위치',
                   '|MDD|≥5%인 단지×면적군 조합만 집계했다(회복률 발산 구간 제외). 자치구당 최소 15개 조합.')


def fig_lollipop(g, reference):
    """MDD로 정렬한 롤리팝 — 순위 자체를 읽어야 할 때 쓴다."""
    fig, ax = plt.subplots(figsize=(10.4, 8.6))
    y = np.arange(len(g))
    ax.hlines(y, g.mdd, 0, color=ec.GRID, linewidth=6, zorder=1)
    colors = [ec.STATUS['good'] if v >= 0 else ec.STATUS['critical'] for v in g.gap]
    ax.scatter(g.mdd, y, s=95, color=ec.CAT[0], zorder=3, edgecolors='white', linewidths=1.5)
    ax.set_yticks(y, [f'{i}  (n={n})' for i, n in zip(g.index, g.n)])
    ax.invert_yaxis()
    ax.axvline(0, color=ec.INK2, linewidth=1)
    ec.grid_x(ax)
    for yi, (mdd, gap) in enumerate(zip(g.mdd, g.gap)):
        ax.text(mdd - 0.4, yi, f'{mdd:.1f}%', va='center', ha='right', fontsize=8.5, color=ec.INK)
        color = ec.STATUS['good'] if gap >= 0 else ec.STATUS['critical']
        sign = '+' if gap >= 0 else ''
        ax.text(0.6, yi, f'전고점 대비 {sign}{gap:.1f}%', va='center', fontsize=8.5,
               color=color, fontweight='bold')
    ax.set_xlim(g.mdd.min() * 1.2, abs(g.mdd.min()) * 0.55)
    ax.set_xlabel('자치구 중위 최대 낙폭 MDD (%)')
    ax.set_title(f'자치구 25곳 낙폭·회복 순위 (기준 {reference})\n'
                f'낙폭 깊은 순 정렬 · 초록 = 전고점 회복, 빨강 = 아직 미회복')
    return ec.save(fig, '24_자치구_롤리팝_순위',
                   '|MDD|≥5%인 조합만 집계했다. n은 자치구별 표본 조합 수다.')


def main():
    g, reference = load_district_table()
    print(f'기준 분기 {reference} · 집계 자치구 {len(g)}개')
    fig_scatter(g, reference)
    fig_lollipop(g, reference)


if __name__ == '__main__':
    main()


def load_district_table_with_recovery():
    """회복률(recovery_ratio)까지 포함한 버전. 막대그래프 전용 — 발산 방지를 위해
    |MDD|>=5%·중위값 규칙은 그대로 유지한다."""
    data = load_csv(DEFAULT_CSV)
    table = axis_inputs(data.quarterly, data.reference, data.partial)
    table = table.merge(data.complexes[['complex_id', 'district']], on='complex_id', how='left')
    solid = table.loc[table.mdd.abs() >= SOLID_MDD].dropna(subset=['recovery_ratio'])
    g = solid.groupby('district').agg(mdd=('mdd', 'median'), recovery=('recovery_ratio', 'median'),
                                      n=('mdd', 'size'))
    g = g.loc[g.n >= MIN_COMBOS].sort_values('mdd')
    return g, str(data.reference)


def fig_butterfly(g, reference):
    """자치구별 MDD와 회복률을 한 막대그래프에 '동시에' 보여준다.

    두 지표는 단위·스케일이 완전히 달라(MDD -25~0%, 회복률 0~400%) 한 축에 같이
    그리면 왜곡된다(대시보드 anti-pattern: 이중 y축 금지). 대신 자치구를 가운데
    공유축으로 두고 좌우로 마주보는 막대(butterfly chart)를 쓴다 — 같은 구를
    나란히 비교하면서도 각 지표는 자기 스케일을 그대로 쓸 수 있다.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 8.6),
                                     gridspec_kw={'wspace': 0.06})
    y = np.arange(len(g))

    ax_l.barh(y, g.mdd, color=ec.CAT[0], height=0.62, zorder=3)
    ax_l.set_xlim(g.mdd.min() * 1.15, 0)
    for yi, v in zip(y, g.mdd):
        ax_l.text(v - g.mdd.min() * 0.02, yi, f'{v:.1f}%', va='center', ha='right',
                 fontsize=9, color=ec.INK)
    ax_l.set_yticks(y, g.index, fontsize=10)
    ax_l.invert_yaxis()
    ec.grid_x(ax_l)
    ax_l.spines['left'].set_visible(False)
    ax_l.set_xlabel('최대 낙폭 MDD 중위값 (%)')
    ax_l.set_title('낙폭', fontsize=11, loc='right')

    colors = [ec.STATUS['warning'] if v < 100 else ec.CAT[2] for v in g.recovery]
    ax_r.barh(y, g.recovery, color=colors, height=0.62, zorder=3)
    ax_r.axvline(100, color=ec.INK2, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
    ax_r.set_xlim(0, g.recovery.max() * 1.15)
    for yi, v in zip(y, g.recovery):
        # 막대 끝이 100% 기준선 바로 앞(80~100)이면 라벨이 그 점선을 관통하므로,
        # 그런 경우만 선 너머로 넘겨 찍는다.
        label_x = 100 + g.recovery.max() * 0.02 if 80 <= v < 100 else v + g.recovery.max() * 0.02
        ax_r.text(label_x, yi, f'{v:.0f}%', va='center', ha='left', fontsize=9, color=ec.INK)
    ax_r.set_yticks(y, [''] * len(g))
    ax_r.invert_yaxis()
    ec.grid_x(ax_r)
    ax_r.spines['right'].set_visible(False)
    ax_r.set_xlabel('하락분 회복률 중위값 (%)')
    ax_r.set_title('회복률', fontsize=11, loc='left')
    ax_r.text(100, -1.3, '100%\n(전고점 회복)', fontsize=8, color=ec.INK2, ha='center', va='bottom')

    fig.suptitle(f'자치구 25곳 · 낙폭과 회복률을 나란히 (기준 {reference})\n'
                f'노란 막대는 회복률 100% 미만(아직 하락분을 다 못 채움)',
                fontsize=13, fontweight='bold', x=0.01, ha='left', y=1.03)
    return ec.save(fig, '25_자치구_버터플라이_MDD_회복률',
                   '|MDD|≥5%인 조합만 집계했다(회복률 발산 구간 제외, 중위값 사용). 두 패널은 각자 스케일을 쓴다 — 이중 y축이 아니다.')


if __name__ == '__main__':
    g2, ref2 = load_district_table_with_recovery()
    fig_butterfly(g2, ref2)
