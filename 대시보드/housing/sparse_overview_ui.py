"""The screen a thinly traded combination gets instead of the full price cards."""
from html import escape

import streamlit as st


def text(value):
    return escape(str(value), quote=True)


def won(amount_manwon: float) -> str:
    """만원 단위 금액을 억·만원으로 끊어 읽는다. 환산가가 아니라 실제 체결 금액이다."""
    billions, rest = divmod(int(round(amount_manwon)), 10000)
    if not billions:
        return f'{rest:,}<small>만원</small>'
    if not rest:
        return f'{billions}<small>억원</small>'
    return f'{billions}억 {rest:,}<small>만원</small>'


def render_sparse_overview(view):
    """Accept only values the calculator could read from the source; never fill a blank."""
    cards = []
    trade = view["latest_trade"]
    if trade:
        cards.append(
            '<article class="sparse-card"><div class="sparse-k">가장 최근에 확인된 거래</div>'
            f'<div class="sparse-v">{won(trade["amount"])}</div>'
            f'<div class="sparse-meta"><b>{text(trade["date"])}</b>'
            f'<span>전용 {trade["area"]:.1f}㎡ · ㎡당 {trade["unit_price"]:,.0f}만원</span></div></article>')

    change = view["change"]
    if change:
        cards.append(
            '<article class="sparse-card"><div class="sparse-k">확인된 기간의 가격 변화</div>'
            f'<div class="sparse-v">{change["pct"]:+,.1f}<small>%</small></div>'
            f'<div class="sparse-meta"><b>{change["from_price"]:,.0f} → {change["to_price"]:,.0f}만원/㎡</b>'
            f'<span>{text(change["from_period"])} → {text(change["to_period"])}</span></div></article>')

    cards.append(
        '<article class="sparse-card"><div class="sparse-k">가격 계산에 사용한 분기</div>'
        f'<div class="sparse-v">{view["qualified"]}<small> / {view["span"]}개 분기</small></div>'
        '<div class="sparse-meta"><b>분기당 거래 3건 이상</b>'
        '<span>나머지는 참고값 또는 거래 없음</span></div></article>')

    st.markdown(
        f'<div class="sparse"><div class="sparse-heading"><div><h2>거래가 있었던 시기의 가격이에요</h2>'
        f'<p>{text(view["subtitle"])}</p></div><span class="badge">{text(view["years"])}</span></div>'
        '<div class="sparse-notice"><div class="sparse-mark">!</div><div>'
        '<h4>아직 가격 흐름을 판단하기엔 거래가 적어요</h4>'
        f'<p>거래가 처음 확인된 <b>{text(view["first_period"])}부터 {view["span"]}개 분기</b> 중, '
        f'거래 3건 이상인 분기는 <b>{view["qualified"]}개</b>예요. '
        '낙폭·회복률은 완결 분기 4개가 필요해 계산하지 않았어요.</p></div></div>'
        '<h3 class="sparse-sub">그래도 확인할 수 있는 정보</h3>'
        f'<p class="sparse-lede">현재 시세가 아니라, 데이터에 실제로 기록된 {text(view["source_label"])}를 보여드려요.</p>'
        f'<div class="sparse-grid">{"".join(cards)}</div>'
        '<p class="sparse-note">여기 금액은 환산가가 아니라 실제 체결 금액이에요. '
        '같은 단지라도 면적·층·동에 따라 다를 수 있어요.</p></div>',
        unsafe_allow_html=True)
