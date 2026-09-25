"""Price overview cards styled after the UI v2 reference image."""
from html import escape

import streamlit as st


def text(value):
    return escape(str(value), quote=True)


def ring(value, label, negative=False):
    shown = "—" if value is None else f"{value:+.1f}%"
    progress = 0 if value is None else min(abs(value), 100)
    color = "#ca6249" if negative else "#13795c"
    return (f'<div class="price-v2-ring" style="--progress:{progress}%;--ring-color:{color}" '
            f'role="img" aria-label="{text(label)} {text(shown)}"><div><strong>{shown}</strong>'
            f'<span>{text(label)}</span></div></div>')


def amount(value, area):
    if value is None or area is None:
        return "자료 부족"
    return f"{value * area / 10000:,.1f}억원"


def price_pair(left_label, left_value, left_period, right_label, right_value, right_period):
    return (f'<div class="price-v2-pair"><div><span>{text(left_label)}</span><strong>{text(left_value)}</strong>'
            f'<small>{text(left_period)}</small></div><span class="price-v2-arrow" aria-hidden="true">➜</span>'
            f'<div><span>{text(right_label)}</span><strong>{text(right_value)}</strong>'
            f'<small>{text(right_period)}</small></div></div>')


def render_price_overview(view):
    """Accept formatted periods and source-derived numeric values from the calculator."""
    area = view["area"]
    area_label = f"{area:g}" if area is not None else "—"
    full_area_label = f"{area:.1f}" if area is not None else "—"
    fall_pair = price_pair("과거 고점", amount(view["peak"], area), view["peak_period"],
                           "고점 이후 저점", amount(view["trough"], area), view["trough_period"])
    now_pair = price_pair("이전 고점", amount(view["peak"], area), view["peak_period"],
                          view["current_label"], amount(view["price"], area), view["current_period"])
    # 기준이 최신 분기에서 물러났으면, 상세를 펼치지 않아도 그 사실이 먼저 보이게 한다.
    stale_note = (f'<p class="price-v2-stale">최신 {text(view["latest_period"])}은 거래 3건에 못 미쳐, '
                  f'마지막으로 3건 이상 거래된 <b>{text(view["current_period"])}</b>을 기준으로 계산했어요.</p>'
                  if view["stale"] else "")
    markup = (f'<div class="price-v2"><div class="price-v2-heading"><div><h2>가격은 이렇게 움직였어요</h2>'
              f'<p>{text(view["subtitle"])}</p></div><span class="badge">{text(view["years"])}</span></div>'
              f'{stale_note}'
              '<div class="price-v2-grid"><article class="price-v2-card">'
              f'<div class="price-v2-top"><div><h3>가장 힘들 때</h3><p>전용 {area_label}㎡로 같은 크기를 비교했습니다.</p></div>'
              f'{ring(view["mdd"], "최대 낙폭(MDD)", True)}</div>{fall_pair}'
              '<div class="price-v2-bottom"><div><span>하락 기간</span>'
              f'<strong>{text(view["decline_duration"])}</strong></div><div><span>가격 기준</span><strong>㎡당 중위가격</strong></div></div></article>'
              '<article class="price-v2-card"><div class="price-v2-top"><div><h3>지금은</h3>'
              f'<p>이전 고점과 현재를 같은 크기로 비교했습니다.</p></div>{ring(view["gap"], "고점 대비 위치")}</div>{now_pair}'
              '<div class="price-v2-bottom"><div><span>고점 이후 현재까지</span>'
              f'<strong>{text(view["since_peak"])}</strong></div><div><span>회복 시점</span>'
              f'<strong>{text(view["recovery_duration"])}</strong></div></div></article>'
              f'<article class="price-v2-card price-v2-evidence"><h3>거래 근거</h3><p>숫자를 만든 {"실제 자료" if view["source_label"] == "실거래" else "예시 자료"}의 양입니다.</p>'
              f'<div class="price-v2-count"><strong>{view["count"]:,}</strong><span>건</span></div>'
              f'<p>{text(view["current_period"])} 기준 가격에 사용한 {text(view["source_label"])}</p>'
              '<dl><div><dt>분기 최소 기준</dt><dd>3건 이상</dd></div>'
              f'<div><dt>계산에 사용한 분기</dt><dd>{view["qualified"]}/{view["total_quarters"]}개</dd></div>'
              f'<div><dt>현재 상태</dt><dd>{text(view["status"])}</dd></div></dl></article></div>'
              '<p class="price-v2-note"><strong>억 단위 금액은 환산가격입니다.</strong> '
              f'선택 조건의 대표면적 {full_area_label}㎡에 각 시점의 ㎡당 중위가격을 곱해 비교했습니다. '
              '같은 크기를 비교하기 위한 값이며 개별 세대의 실제 거래금액은 면적·층·동에 따라 다를 수 있습니다.</p>'
              f'<details class="price-v2-method"><summary>산출 기준 확인</summary><p>{text(view["method_note"])}</p></details></div>')
    st.markdown(markup, unsafe_allow_html=True)
