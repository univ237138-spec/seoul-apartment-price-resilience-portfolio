"""Distribution tab: source-derived scatter plot and a plain-language reading guide."""
from html import escape

import plotly.graph_objects as go
import streamlit as st

from .distribution import distribution_narrative, distribution_values
from .models import quarter_label
from .ui import h, html


def distribution_chart(view) -> go.Figure:
    points, own = view["points"], view["selected"]
    fig = go.Figure()
    if points.empty:
        return fig
    # Include every finite observation, even outliers. The horizontal reference is always 0%.
    xmin, xmax = float(points.mdd.min()), float(points.mdd.max())
    xpad = max((xmax - xmin) * .14, 2)
    xr = [xmin - xpad, min(0, xmax + xpad)]
    ymin, ymax = min(float(points.gap.min()), 0), max(float(points.gap.max()), 0)
    ypad = max((ymax - ymin) * .2, 6)
    yr = [ymin - ypad, ymax + ypad]
    divider = view["median_mdd"]
    if divider is not None:
        fig.add_shape(type="rect", x0=divider, x1=xr[1], y0=0, y1=yr[1],
                      fillcolor="rgba(22,114,84,.055)", line_width=0, layer="below")
        fig.add_vline(x=divider, line_width=1, line_dash="dash", line_color="#aabdb5")
        fig.add_annotation(x=divider, y=1.02, xref="x", yref="paper", showarrow=False,
                           text=f'비교 단지 낙폭 중앙값 {divider:.1f}%',
                           font={"size": 10, "color": "#738680"})
        for x, y, label, anchor in [
            (.02, .97, "낙폭 크지만 전고점 회복", "left"),
            (.98, .97, "낙폭 작고 전고점 회복", "right"),
            (.02, .03, "낙폭 크고 전고점 미회복", "left"),
            (.98, .03, "낙폭 작지만 전고점 미회복", "right"),
        ]:
            fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=label,
                               showarrow=False, xanchor=anchor, font={"size": 10, "color": "#879b92"})
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="#7a9d8d")

    for selected, label, color in [(False, "비교 단지", "#b6c3bf"), (True, "선택 단지", "#13795c")]:
        subset = points.loc[points.complex_id.eq(view["selected_id"]) == selected]
        if subset.empty:
            continue
        custom = [[escape(str(row["name"])), escape(str(row["address"])), quarter_label(row.basis),
                   int(row["count"]), " · 부분 분기 잠정값" if row.partial else "", row.complex_id]
                  for _, row in subset.iterrows()]
        if selected:
            fig.add_trace(go.Scatter(x=subset.mdd.tolist(), y=subset.gap.tolist(), mode="markers",
                                    marker={"size": 31, "color": "rgba(19,121,92,.17)"},
                                    hoverinfo="skip", showlegend=False))
        show_names = len(points) <= 15 and not selected
        fig.add_trace(go.Scatter(
            x=subset.mdd.tolist(), y=subset.gap.tolist(), name=label,
            mode="markers+text" if show_names else "markers",
            text=[escape(str(name)) for name in subset.name] if show_names else None,
            textposition="top center", textfont={"size": 11, "color": "#315c4d"},
            marker={"size": 18 if selected else 12, "color": color, "line": {"color": "white", "width": 2}},
            customdata=custom,
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>최대 낙폭(MDD) %{x:.1f}%"
                           "<br>고점 대비 위치 %{y:+.1f}%<br>기준: %{customdata[2]}%{customdata[4]}"
                           "<br>기준 분기 거래 %{customdata[3]}건<extra></extra>"),
        ))
    if own is not None:
        fig.add_annotation(x=own["mdd"], y=own["gap"], text=escape(str(own["name"])),
                           showarrow=True, arrowhead=0, arrowwidth=1.5, arrowcolor="#13795c",
                           ax=-35 if own["mdd"] > (xr[0] + xr[1]) / 2 else 35, ay=-35,
                           bgcolor="rgba(255,254,250,.92)", borderpad=3,
                           font={"size": 13, "color": "#13795c"})
    fig.update_layout(
        # 오른쪽 설명 칼럼과 카드 높이를 맞춘 값(1440px 기준 측정: 좌 597 / 우 722, 차이 125).
        height=585, margin={"l": 12, "r": 18, "t": 45, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafcf9",
        font={"family": "Malgun Gothic, sans-serif", "size": 11, "color": "#738680"},
        xaxis={"range": xr, "title": {"text": "최대 낙폭(MDD, %) · 오른쪽일수록 덜 하락", "font": {"size": 11}},
               "tickformat": ".0f", "ticksuffix": "%", "gridcolor": "#e3ebe6", "zeroline": False},
        yaxis={"range": yr, "title": {"text": "고점 대비 위치(%)", "font": {"size": 11}},
               "tickformat": "+.0f", "ticksuffix": "%", "gridcolor": "#e3ebe6", "zeroline": False},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.16},
        hoverlabel={"bgcolor": "#18372e", "font": {"color": "white"}},
    )
    return fig


def render_distribution(data, selected):
    view = distribution_values(data, selected)
    headline, explanation = distribution_narrative(view)
    html('<div class="distribution-heading"><div><div class="eyebrow">단지 비교</div>'
         '<h2>우리 단지는 덜 떨어지고, 더 회복했을까요?</h2>'
         '<p>같은 자치구·같은 면적군의 단지를 최대 낙폭과 현재 가격 위치로 비교했어요.</p></div>'
         f'<span class="badge">{h(view["district"])} · {h(view["area_group"])}</span></div>')
    if data.source == "demo":
        st.caption("시연용 가격 흐름으로 계산한 예시 분포예요. 실제 단지 평가가 아니에요.")
    left, right = st.columns([7, 3], gap="medium")
    with left:
        with st.container(key="distribution-chart-card"):
            html('<div class="chart-heading"><h3>가격 방어력 × 고점 회복 위치</h3></div>')
            if view["points"].empty:
                st.info("이 구·면적 조건에서는 분포를 그릴 수 있는 단지가 없어요. 다른 조건을 선택해 주세요.")
            else:
                st.plotly_chart(distribution_chart(view), width="stretch", config={"displayModeBar": False},
                                key=f'distribution-{selected.source}-{selected.complex_id}-{selected.area_group}')
                if view["selected"] is not None:
                    own = view["selected"]
                    html(f'<div class="distribution-selected"><b>{h(own["name"])}</b>'
                         f'<span>최대 낙폭 {own["mdd"]:.1f}% · 고점 대비 {own["gap"]:+.1f}%</span></div>')
                st.caption("점에 마우스를 올리거나 터치하면 단지명·실제 수치·기준 분기를 확인할 수 있어요. 겹친 점은 아래 표에서도 확인할 수 있어요.")
                if view["selected"] is None:
                    st.info(explanation)
    with right:
        html(f'<div class="distribution-insights"><article class="distribution-verdict"><div class="eyebrow">한눈에 보면</div>'
             f'<h3>{h(headline)}</h3><p>{h(explanation)}</p></article>'
             '<article class="card"><h3>어떻게 읽나요?</h3>'
             '<p><b>오른쪽</b>일수록 최대 낙폭이 작고, <b>위쪽</b>일수록 이전 고점 대비 현재 위치가 높아요. '
             '0%는 이전 고점 회복, 0% 위는 전고점 돌파예요.</p>'
             '<p>세로 점선은 선택 단지를 제외한 비교 단지의 낙폭 중앙값이에요. 배경색은 그보다 낙폭이 작고 전고점을 회복한 영역이에요.</p></article>'
             '<article class="card"><h3>비교 기준</h3>'
             f'<p>같은 구·면적 {view["candidates"]:,}곳 중 <b>{len(view["points"]):,}곳 표시</b><br>'
             f'기록 부족 {view["insufficient"]:,}곳 · 하락 구간 없음 {view["no_decline"]:,}곳 제외</p>'
             '<p>거래 3건 이상인 완결 분기 4개 이상이 필요해요. 최신 부분 분기도 거래 3건 이상이면 포함해요. '
             '하락 구간이 없는 단지는 고점 대비 위치를 계산하지 않아 제외해요.</p>'
             f'<p>{view["stale_count"]:,}곳은 최신 분기의 거래가 부족해 마지막 유효 분기를 사용해요. '
             '단지마다 기준 시점과 관측 기간이 다를 수 있어요.</p>'
             '<p>상단 등급은 전체 지역의 같은 면적군 기준이며, 이 분포의 비교 범위와 달라요.</p></article></div>')
    if not view["points"].empty:
        with st.expander("분포에 표시된 단지와 기준 시점 보기"):
            table = view["points"][["name", "dong", "address", "mdd", "gap", "basis", "count", "partial"]].copy()
            table["basis"] = table.basis.map(quarter_label)
            table["partial"] = table.partial.map({True: "잠정", False: "완결"})
            table.columns = ["단지명", "동", "주소", "최대 낙폭(MDD, %)", "고점 대비 위치(%)", "기준 분기", "거래건수", "분기 상태"]
            st.dataframe(table, hide_index=True, width="stretch")
