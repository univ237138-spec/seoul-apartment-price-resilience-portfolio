"""HTML presentation only; all displayed values come from shared snapshots."""
from html import escape

import pandas as pd
import streamlit as st

from .metrics import narrative
from .models import Snapshot, duration, quarter_label
from .scoring import DEFENSE_WEIGHT, RECOVERY_WEIGHT, GRADE_BANDS
from .visuals import facility_map, price_chart
from .facility_icons import STYLES, icon_html


def h(value):
    return escape(str(value), quote=True)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def josa_i_ga(word: str) -> str:
    """이/가 조사. 한글 마지막 글자의 받침 유무로 고른다(음절 코드포인트가 0xAC00부터
    28개 종성 단위로 배열돼, 나머지가 0이면 받침이 없다는 성질을 이용한다)."""
    if not word or not ("가" <= word[-1] <= "힣"):
        return "가"
    return "가" if (ord(word[-1]) - 0xAC00) % 28 == 0 else "이"


def fmt(metric, digits=0, suffix="", missing="자료 부족", signed=False):
    if metric.value is None:
        if metric.status == "no_decline":
            return "하락 구간 없음"
        return missing
    return f'{metric.value:{"+" if signed else ""},.{digits}f}{suffix}'


def section(title, subtitle="", badge=""):
    html(f'<div class="section-heading"><div><h2>{h(title)}</h2><p>{h(subtitle)}</p></div>'
         f'<span class="badge">{h(badge)}</span></div>')


def recovery_span(metric) -> str:
    """저점에서 전고점까지 걸린 기간. 분기 수보다 년·개월이 바로 읽힌다."""
    if metric.value is not None:
        return duration(metric.value)
    return {"unrecovered": "아직 미회복", "no_decline": "하락 구간 없음"}.get(metric.status, "자료 부족")


def score_details(s: Snapshot) -> str:
    """Native, keyboard-operable disclosure using the exact population used for scoring."""
    score = s.score
    context = f'{s.info.get("district", "")} · {s.area_group}'.strip(" ·")
    if not score:
        body = ('<h3>등급 산출에 필요한 기록이 부족해요</h3>'
                '<p>분기당 거래 3건 이상인 완결 분기가 4개 이상 필요해요. '
                '최신 분기의 거래가 적으면 마지막으로 3건 이상 거래된 분기를 기준으로 계산해요.</p>')
    elif s.source == "demo":
        body = (f'<h3>{score["total"]}점 · 시연용 점수예요</h3>'
                f'<p>방어 {score["defense"]}/{DEFENSE_WEIGHT}점 + 회복 {score["recovery"]}/{RECOVERY_WEIGHT}점.</p>'
                '<p>시연 화면은 미리 정한 예시 점수를 사용해요. 실제 비교 집단·백분위·등수는 '
                '실데이터 화면에서 느낌표를 눌러 확인해 주세요.</p>')
    else:
        dp, rp, percentile = (score[k] for k in ["defense_percentile", "recovery_percentile", "percentile"])
        count = score["axis_peers"]
        if score["no_decline"]:
            defense_note = '관측 기간에 하락 구간이 없어 방어 점수를 만점으로 부여해요.'
            recovery_note = '하락 구간이 없어 고점 대비 위치는 계산하지 않고 회복 점수를 만점으로 부여해요.'
        else:
            defense_note = (f'이 아파트 낙폭 <b>{h(fmt(s.metrics["mdd"], 1, "%"))}</b> · '
                            f'같은 면적군 {count:,}곳에서 낙폭 방어 <b>{dp:.1f}백분위</b>예요.')
            recovery_note = (f'전고점 대비 <b>{h(fmt(s.metrics["gap"], 1, "%", signed=True))}</b> · '
                             f'같은 면적군 {count:,}곳에서 고점 대비 위치 <b>{rp:.1f}백분위</b>예요.')
        axes = ""
        for label, key, weight, value, description, start, end in [
            ("낙폭 방어", "defense", DEFENSE_WEIGHT, dp, defense_note, "많이 빠짐", "안 빠짐"),
            ("고점 대비 위치", "recovery", RECOVERY_WEIGHT, rp, recovery_note, "전고점 못 미침", "전고점 돌파"),
        ]:
            axes += (f'<section class="score-axis"><div class="score-axis-heading"><b>{label}</b>'
                     f'<strong>{score[key]} / {weight}점</strong></div><p>{description}</p>'
                     f'<div class="score-axis-track {key}" role="meter" aria-label="{label} 백분위" '
                     f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{value:.2f}">'
                     f'<span style="width:{value:.4f}%"></span></div>'
                     f'<div class="score-axis-labels"><span>{start}</span><span>{end}</span></div></section>')
        bands = list(reversed(GRADE_BANDS))
        scale = ""
        for i, (floor, label) in enumerate(bands):
            ceiling = bands[i + 1][0] if i + 1 < len(bands) else 100
            interval = f'{floor} 이상 {ceiling} 미만' if ceiling < 100 else f'{floor} 이상 100 이하'
            selected = ' selected' if label == score["grade"] else ''
            scale += (f'<span class="grade-band{selected}" style="flex:{ceiling - floor}" '
                      f'title="{label}: {interval} 백분위">{label}</span>')
        maximum = score["recovery_max"]
        evidence = (f'현재 데이터에서 산출 가능한 전체 면적군 {score["recovery_sample_count"]:,}개 '
                    f'단지·면적 조합의 하락분 회복률 최댓값은 <b>{maximum:,.2f}%</b>예요. '
                    if maximum is not None else '현재 데이터에는 하락분 회복률을 계산할 하락 구간이 없어요. ')
        partial = ' · 부분 분기 잠정값' if s.metrics["price"].status == "partial" else ''
        body = (f'<div class="score-detail-heading"><h3>{score["total"]}점은 이렇게 계산했어요</h3>'
                f'<span>{h(context)}</span></div>'
                f'<p>총점 = <b>낙폭 방어 백분위 × {DEFENSE_WEIGHT}% + 고점 대비 위치 백분위 × {RECOVERY_WEIGHT}%</b>. '
                '두 축 모두 같은 면적군의 다른 단지와 비교한 상대 순위예요.</p>'
                f'<p class="score-population">비교 범위: 데이터 전체 지역 · {h(s.area_group)}. 자치구로 제한하지 않아요.</p>'
                f'{axes}<div class="score-peer-rank">같은 면적군 {score["peers"]:,}곳 중 '
                f'<b>{score["rank"]:,}위</b></div>'
                f'<div class="grade-scale" aria-label="등급 구간, 현재 {h(score["grade"])}">{scale}</div>'
                f'<div class="grade-position" style="--position:{percentile:.4f}%"><span>▲</span></div>'
                f'<div class="grade-position-label">이 아파트 · 상위 {100 - percentile:.1f}% '
                f'({percentile:.1f}백분위)</div>'
                '<div class="score-grade-key">등급 기준(백분위): C 0 · C+ 15 · B 35 · B+ 55 · A 75 · A+ 90 이상</div>'
                '<div class="score-rationale"><b>왜 회복률(%) 대신 고점 대비 위치를 쓰나요?</b>'
                '<p>하락분 회복률은 낙폭이 작으면 분모(고점 − 저점)가 0에 가까워져 값이 크게 튈 수 있어요. '
                f'{evidence}고점 대비 위치는 하락폭으로 나누지 않아 이 영향을 줄여줘요.</p></div>'
                '<details class="score-method-notes"><summary>동점 처리·표본·계산 시점 자세히 보기</summary>'
                '<p>백분위 = (더 낮은 값의 수 + 같은 값의 수 × 0.5) ÷ 비교 수 × 100. '
                '자기 자신도 포함하며, 백분위가 높을수록 비교 집단에서 높은 위치예요. '
                '하락이 없는 단지는 두 축 만점으로 처리하고 축별 비교에서는 제외하지만, 전체 등수·등급에는 포함해요.</p>'
                '<p>등수는 반올림 전 총점의 내림차순이며 동점은 공동 순위예요. 등급과 상위 비율은 총점 백분위로 정해요. '
                '각 점수와 총점을 따로 반올림해 표시하므로 합계가 1점 다를 수 있어요. 20/80 가중치는 서비스에서 정한 기준이에요.</p>'
                f'<p>이 단지 기준: {h(quarter_label(s.basis))}{partial}. 분기별 ㎡당 중위가격을 사용하며, '
                '거래 3건 이상인 완결 분기 4개 이상이 필요해요. 단지마다 마지막 유효 분기가 다를 수 있어요.</p></details>'
                '<p class="score-disclaimer">등급은 절대 기준이 아니라 같은 조건의 단지 대비 상대 순위예요. '
                '거래가 새로 쌓이면 함께 바뀔 수 있어요.</p>')
    return ('<details class="score-details"><summary aria-label="등급 산출 기준" title="등급 산출 기준 열기·닫기">'
            '<span aria-hidden="true">!</span></summary><div class="score-popover">'
            f'{body}</div></details>')


def render_summary(s: Snapshot):
    n, m = narrative(s), s.metrics
    if s.score:
        score = s.score
        ring = f'<div class="score-ring" style="--score:{score["total"]}%"><div><strong>{h(score["grade"])}</strong><b>{score["total"]}점</b></div></div>'
        right = [f'방어 {score["defense"]}/{DEFENSE_WEIGHT}', f'회복 {score["recovery"]}/{RECOVERY_WEIGHT}', n["badge"]]
        note = ""
    else:
        ring = '<div class="score-ring pending" style="--score:0%"><div><strong>—</strong><b>등급 산출 불가</b></div></div>'
        right = ["표본 부족", "표본 부족", n["badge"]]
        note = "거래 3건 이상인 완결 분기 4개가 필요해요"
    left = [f'최대 낙폭(MDD) {fmt(m["mdd"], 1, "%")}', f'고점 대비 위치 {fmt(m["gap"], 1, "%", signed=True)}',
            f'회복 기간 {recovery_span(m["recovery_quarters"])}']
    score_rows = ''.join(f'<div class="score-row"><span>{h(a)}</span><b>{h(b)}</b></div>' for a, b in zip(left, right))
    score_note = f'<small>{h(note)}</small>' if note else ""
    headline = h(n["headline"]).replace("\n", "<br>")
    html('<div class="score-help-hint">아래 ! 아이콘을 누르면 등급 산출 기준이 열려요</div>'
         f'<div class="summary-grid"><article class="card conclusion"><div class="conclusion-copy">'
         f'<div class="eyebrow">이 아파트의 한줄 결론</div><h1>{headline}</h1><p>{h(n["summary"])}</p></div>'
         f'<div class="score-panel"><div class="score-ring-wrap">{ring}{score_details(s)}</div>'
         f'<div class="score-rows">{score_rows}</div>{score_note}</div></article>'
         f'<aside class="card check-card"><div class="eyebrow">구매 전 체크</div><h3>{h(n["check_title"])}</h3><p>{h(n["check"])}</p>'
         '<div class="check-foot">가격 회복력은 과거 기록을 비교하는 정보예요.</div></aside></div>')


def render_metrics(s: Snapshot):
    m = s.metrics
    fall = f'{abs(m["mdd"].value):.1f}% 하락' if m["mdd"].value is not None else "장기 자료 부족"
    current = f'이전 고점 {m["gap"].value:+.1f}%' if m["gap"].value is not None else "표본 가격 확인"
    price_name = "표본 최근 분기 가격" if s.partial else "기준 분기 가격"
    period = quarter_label(s.reference)
    part = " · 부분 표본" if s.partial else ""
    count = fmt(m["quarter_count"], suffix="건 거래")
    peak_date = quarter_label(m["peak"].period) if m["peak"].value is not None else "관측 기간 부족"
    trough_date = quarter_label(m["trough"].period) if m["trough"].value is not None else "관측 기간 부족"
    if m["mdd"].status == "no_decline" or m["peak"].status == "no_decline":
        peak_date = trough_date = "하락 구간 없음"
    html(f'<div class="metric-grid">'
         f'<article class="card metric-card"><div class="metric-icon">↓</div><div class="metric-label">가장 힘들 때</div><h2>{h(fall)}</h2>'
         f'<div class="metric-detail"><div><span>고점</span><b>{h(fmt(m["peak"], suffix="만원/㎡"))}</b><small>{h(peak_date)}</small></div>'
         f'<div><span>저점</span><b>{h(fmt(m["trough"], suffix="만원/㎡"))}</b><small>{h(trough_date)}</small></div></div></article>'
         f'<article class="card metric-card"><div class="metric-icon">↗</div><div class="metric-label">지금은</div><h2>{h(current)}</h2>'
         f'<div class="metric-detail"><div><span>{price_name}</span><b>{h(fmt(m["price"], suffix="만원/㎡", missing="해당 분기 거래 없음"))}</b></div>'
         f'<div><span>기준 시점</span><b>{h(period)}</b><small>{h(part)}</small></div></div></article>'
         f'<article class="card metric-card"><div class="metric-icon">✓</div><div class="metric-label">믿고 볼 수 있나요?</div><h2>{h(count)}</h2>'
         f'<div class="metric-detail"><div><span>최근 1년</span><b>{h(fmt(m["year_count"], suffix="건"))}</b></div>'
         f'<div><span>{"관측 분기 표본" if s.partial else "기준 분기"}</span><b>{h(fmt(m["quarter_count"], suffix="건"))}</b></div></div></article></div>')


def render_chart(s: Snapshot, period_range=None):
    n = narrative(s)
    left, right = st.columns([7, 3], gap="medium")
    with left:
        with st.container(key="price-chart-card"):
            # 낙폭을 못 낸 조합에 "떨어진 후"를 묻는 건 답이 없는 질문이라, 제목도 같이 바꾼다.
            heading = ("떨어진 후 어디까지 돌아왔을까요?" if s.metrics["mdd"].value is not None
                       else "거래가 확인된 시점만 볼게요")
            axis_key = f"axis-zero-{s.source}-{s.complex_id}-{s.area_group}"
            zero_start = bool(st.session_state.get(axis_key, False))
            figure = price_chart(s, period_range, zero_start=zero_start)
            zero_omitted = figure.layout.yaxis.range[0] > 0
            axis_badge = "Y축 자동 확대 · 0 생략" if zero_omitted else "Y축 0부터 표시"
            html(f'<div class="chart-heading"><h3>{h(heading)}</h3><div class="chart-heading-badges">'
                 f'<span class="axis-badge">{h(axis_badge)}</span><span class="badge">{h(n["badge"])}</span></div></div>')
            st.toggle("Y축 0부터 보기", key=axis_key,
                      help="켜면 가격 변화가 작아 보일 수 있지만 0만원/㎡부터 전체 높이로 비교할 수 있어요.")
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False}, key=f"chart-{s.source}-{s.complex_id}")
            if zero_omitted:
                st.caption("가격 변화를 읽기 쉽게 Y축을 관측값 범위에 맞춰 확대했어요. 0부터 시작하지 않으며, 왼쪽 아래의 ‘// 0 생략’이 이를 표시해요.")
            else:
                st.caption("Y축을 0만원/㎡부터 표시하고 있어요. 확대해서 보려면 ‘Y축 0부터 보기’를 끄세요.")
            if s.series.price.notna().sum() <= 1:
                st.caption("관측된 분기가 1개라 점으로 표시했어요. 현재 시세가 아닌 CSV 표본 가격이에요.")
            elif s.partial:
                st.caption("현재 분기는 부분 분기 잠정값이에요. 거래 3건 미만은 참고값이며, 거래가 없는 분기는 연결하지 않아요.")
            elif s.metrics["mdd"].value is None:
                st.caption("가격 점은 관측값이에요. 최소 분기·거래건수 조건을 충족해야 회복 지표를 계산해요.")
            else:
                st.caption("고점·저점은 최대 낙폭 구간 기준이에요. 거래 3건 미만은 참고값이며, 빈 분기는 연결하지 않아요.")
    with right:
        html(f'<div class="card insights"><div class="insight positive"><h4>좋았던 점</h4><p>{h(n["positive"])}</p></div>'
             f'<div class="insight watch"><h4>살펴볼 점</h4><p>{h(n["watch"])}</p></div></div>')
    with st.expander("하락분 회복률과 산출 기준 보기"):
        cols = st.columns(3)
        cols[0].metric("하락분 회복률", fmt(s.metrics["recovery"], 1, "%"))
        cols[1].metric("고점 대비 기준 가격", fmt(s.metrics["gap"], 1, "%", signed=True))
        span = s.metrics["recovery_quarters"]
        cols[2].metric("저점 이후 고점 회복까지", recovery_span(span))
        st.caption("하락분 회복률 = (기준 가격 − 저점) ÷ (고점 − 저점) × 100. 100%를 넘으면 하락분 이상을 회복한 상태예요.")
        st.caption("고점 대비 위치 = (기준 가격 ÷ 고점 − 1) × 100. 두 비율은 서로 다른 지표예요.")
        if s.metrics["mdd"].reason:
            st.info(s.metrics["mdd"].reason)
        table = s.series[["quarter", "price", "count"]].copy()
        table["quarter"] = table["quarter"].astype(str)
        table.columns = ["분기", "㎡당 중위가격(만원)", "거래건수"]
        st.dataframe(table, hide_index=True, width="stretch")


def comparison_details(s: Snapshot) -> str:
    """One native disclosure per card; all facts belong to this snapshot."""
    facts = []
    for key, label, unit in [("built_year", "건축연도", "년"), ("households", "세대수", "세대"),
                             ("parking_ground", "지상 주차대수", "대"),
                             ("parking_underground", "지하 주차대수", "대"),
                             ("corridor_type", "복도유형", "")]:
        value = s.info.get(key)
        if value is None or pd.isna(value) or str(value).strip() in ("", "-"):
            text = "자료 없음"
        elif key == "built_year":
            text = f'{value:.0f}년'
        elif unit:
            text = f'{value:,.0f}{unit}'
        else:
            text = str(value)
        facts.append(f'<div class="complex-fact"><dt>{label}</dt><dd>{h(text)}</dd></div>')

    facilities = []
    for _, facility in s.facilities.iterrows():
        style = STYLES[facility.kind]
        if facility.status == "ok":
            name, distance = facility["name"], f'{facility.distance:,.0f}m'
        else:
            name = {"no_coordinates": "위치 정보 없음", "unmapped": "1km 내 매핑된 시설 없음",
                    "invalid": facility["name"] or "시설 정보 확인 필요"}.get(facility.status, "자료 없음")
            distance = "1km 밖" if facility.status == "invalid" else ""
        facilities.append(f'<div class="compare-facility" style="--facility-color:{style["color"]}">'
                          f'<div class="compare-facility-kind">{icon_html(facility.kind)}<b>{style["label"]}</b></div>'
                          f'<div class="compare-facility-info"><span>{h(name)}</span><strong>{h(distance)}</strong></div></div>')
    note = "시연용 시설 예시이며 단지 기본 정보는 제공하지 않아요." if s.source == "demo" else "시설은 거래 당시 현황과 다를 수 있어요."
    return (f'<details class="compare-details"><summary aria-label="{h(s.info["name"])} 단지 특성 더보기">'
            '<span class="details-closed">더보기</span><span class="details-open">접기</span>'
            '<span class="details-chevron" aria-hidden="true">⌄</span></summary>'
            '<div class="compare-details-content"><h4>단지 기본 정보</h4>'
            f'<dl class="complex-facts">{"".join(facts)}</dl><h4>최근접 생활시설</h4>'
            f'<p class="compare-detail-note">직선거리 기준 · 유형별 최근접 시설</p>{"".join(facilities)}'
            f'<p class="compare-detail-note">{note}</p></div></details>')


def render_comparison(snapshots: list[Snapshot], selected: str):
    cards = []
    for s in snapshots:
        m = s.metrics
        gap = fmt(m["gap"], 1, "%", signed=True)
        rows = [("가장 크게 하락", fmt(m["mdd"], 1, "%")),
                ("지금 위치", f"고점 {gap}" if m["gap"].value is not None else "장기 자료 부족"),
                ("표본 ㎡당 가격" if s.partial else "기준 ㎡당 가격", fmt(m["price"], suffix="만원", missing="해당 분기 거래 없음"))]
        body = ''.join(f'<div class="compare-row {"compare-position" if i == 1 else ""}"><span>{h(label)}</span><b>{h(value)}</b></div>'
                       for i, (label, value) in enumerate(rows))
        cards.append(f'<article class="card compare-card {"selected" if s.complex_id == selected else ""}">'
                     f'<span class="muted">{h(s.info["dong"])}</span><h3>{h(s.info["name"])}</h3>{body}{comparison_details(s)}</article>')
    html('<div class="compare-grid">' + ''.join(cards) + '</div>')


def render_facilities(s: Snapshot):
    items, available = [], []
    for _, f in s.facilities.iterrows():
        kind = f.kind
        title = STYLES[kind]["label"]
        facility_name = ""
        if f.status == "ok":
            distance = f'{f.distance:,.0f}m'
            facility_name = f["name"]
            if s.source == "demo" and pd.notna(f["count"]):
                body = f'{int(f.count_radius)}km 안에 <b>{int(f["count"])}{ "개" if kind == "학교" else "곳"}</b>'
                sub = "시연용 시설 개수"
            else:
                body = f'가까운 {"점포" if kind == "대규모점포" else "역" if kind == "지하철" else title} <b>{distance}</b>'
                sub = "개수는 집계자료 없음" if kind in ["학교", "대규모점포"] else ""
            available.append(f'{title} {distance}')
        elif f.status == "invalid":
            # 이름·좌표는 있지만 1km를 넘는 경우: 거리는 숨기되 이유를 사실 그대로 말한다.
            facility_name = f["name"]
            body = f'가장 가까운 {title}{josa_i_ga(title)} 1km 밖이에요'
            sub = "집계자료 없음" if kind in ["학교", "대규모점포"] else ""
        else:
            body = {"no_coordinates": "위치 정보 없음", "unmapped": "1km 내 매핑된 시설 없음"}.get(f.status, "자료 없음")
            sub = "집계자료 없음" if kind in ["학교", "대규모점포"] else ""
        name_html = f'<div class="facility-name">{h(facility_name)}</div>' if facility_name else ""
        items.append(f'<div class="facility-item" style="--facility-color:{STYLES[kind]["color"]}">'
                     f'<h4>{icon_html(kind)}{title}</h4>{name_html}<p>{body}</p><small>{h(sub)}</small></div>')
    summary = "지하철과 공원이 가깝고, 학교와 장보기 시설도 고르게 갖춰져 있어요." if s.source == "demo" else (
        "확인된 최근접 시설은 " + ", ".join(available) + "예요." if available else "위치와 시설 자료를 확보하면 주변 생활시설을 함께 확인할 수 있어요.")
    left, right = st.columns(2, gap="medium")
    with left:
        html('<div class="card facilities-card"><div class="facility-grid">' + ''.join(items) + '</div>'
             f'<div class="life-summary"><h4>생활권 한줄 {"평가 · 예시" if s.source == "demo" else "정보"}</h4><p>{h(summary)}</p></div></div>')
    with right:
        with st.container(key="facility-map-card"):
            html(f'<div class="map-apartment" style="--facility-color:{STYLES["아파트"]["color"]}">'
                 f'{icon_html("아파트")}<div><small>기준 아파트</small><strong>{h(s.info["name"])}</strong></div></div>')
            if s.source == "demo":
                demo_pins = []
                positions = {"지하철": "pin-transit", "학교": "pin-school", "공원": "pin-park", "대규모점포": "pin-shop"}
                for _, f in s.facilities.iterrows():
                    if f.status == "ok":
                        demo_pins.append(f'<div class="map-pin {positions[f.kind]}" style="--facility-color:{STYLES[f.kind]["color"]}">'
                                         f'{icon_html(f.kind)}<span>{h(f["name"])}</span></div>')
                html(f'<div class="demo-map" role="img" aria-label="시연용 시설 배치 예시">'
                     '<div class="map-road road-one"></div><div class="map-road road-two"></div><div class="map-radius"></div>'
                     f'<div class="map-pin map-home" style="--facility-color:{STYLES["아파트"]["color"]}">'
                     f'{icon_html("아파트")}<span>{h(s.info["name"])}</span></div>{"".join(demo_pins)}'
                     '<span class="map-scale">생활시설 배치 예시</span></div>')
                st.caption("시설 위치와 거리는 시연용 예시예요. 실제 지도나 실제 입지 평가가 아니에요.")
            else:
                deck = facility_map(s)
                if deck is None:
                    html('<div class="map-empty"><span>⌖</span><h3>위치 정보가 아직 없어요</h3><p>좌표가 있는 단지를 선택하면 주변 시설을 지도에서 확인할 수 있어요.</p></div>')
                else:
                    st.pydeck_chart(deck, height=420, key=f"map-{s.complex_id}")
                st.caption("원은 반경 1km · 유형별 최근접 시설만 표시 · 직선거리 기준")
            legend = ''.join(f'<span style="--facility-color:{style["color"]}">{icon_html(kind)}{style["label"]}</span>'
                             for kind, style in STYLES.items())
            html(f'<div class="map-legend" aria-label="지도 아이콘 범례">{legend}</div>')
    html('<div class="facility-footnote">시설은 거래 당시 현황과 다를 수 있어요. 배경지도가 연결되지 않아도 시설명과 거리는 위 카드에서 확인할 수 있어요.</div>')


def render_data_notes(data, s):
    with st.expander("데이터 범위와 단지 정보"):
        st.write(f'관측 범위: {data.start:%Y.%m.%d} ~ {data.end:%Y.%m.%d} · {s.area_group}')
        st.caption("대표가격은 거래별 ㎡당 가격의 분기 중위값이에요. 가격·거래건수는 제공된 데이터 범위에 한정돼요.")
        if data.source == "real":
            st.caption("최신 관측 분기를 현재·잠정 기준으로 사용해요. 장기 지표는 분기당 3건 이상인 완결 분기 4개 이상과 현재 분기 3건 이상이 필요해요. 조건을 충족하면 현재 부분 분기도 잠정값으로 계산에 포함해요.")
            st.write(s.info["address"])
            details = []
            for key, label, unit in [("households", "세대수", "세대"), ("built_year", "건축년도", "년")]:
                value = s.info.get(key)
                if pd.notna(value):
                    details.append(f"{label} {value:,.0f}{unit}")
            if details:
                st.write(" · ".join(details))
            if s.info.get("bundle_conflict"):
                st.caption("동일 단지의 정보 묶음이 서로 달라 가장 많이 관측된 묶음을 사용했어요. 동률이면 원본에서 먼저 나온 값을 사용해요.")
        st.dataframe(pd.DataFrame(data.diagnostics.items(), columns=["항목", "확인 결과"]).astype(str), hide_index=True, width="stretch")
    html('<footer class="page-footer"><b>H</b><span>우리집 가격체력</span><p>실거래 이력으로 가격 방어와 회복 특성을 비교해요.<br>과거의 회복이 미래의 가격을 보장하지는 않아요.</p></footer>')
