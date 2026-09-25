"""Run: python -m streamlit run 대시보드/streamlit_app.py"""
import os
from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from housing.data import DEFAULT_CSV, PIPELINE_VERSION, DataError, load_csv
from housing.demo import DEMO_VERSION, demo_dataset
from housing.metrics import build_snapshot
from housing.models import AREA_LABELS, quarter_label
from housing.price_overview import overview_values
from housing.price_overview_ui import render_price_overview
from housing.sparse_overview import sparse_values
from housing.sparse_overview_ui import render_sparse_overview
from housing.distribution_ui import render_distribution
from housing.ui import (h, html, section, render_summary, render_chart,
                        render_comparison, render_facilities, render_data_notes)


st.set_page_config(page_title="우리집 가격체력", page_icon="🏡", layout="wide", initial_sidebar_state="collapsed")
st.html(f'<style>{(APP_DIR / "style.css").read_text(encoding="utf-8")}</style>')
st.html(f'<style>{(APP_DIR / "price_overview.css").read_text(encoding="utf-8")}</style>')


@st.cache_data(show_spinner=False)
def cached_dataset(mode, path, modified_ns, size, demo_version):
    return demo_dataset() if mode == "demo" else load_csv(path)


def choices(data, district, area):
    ids = data.quarterly.loc[data.quarterly.area_group == area, "complex_id"].unique()
    return data.complexes.loc[(data.complexes.district == district) & data.complexes.complex_id.isin(ids)].sort_values(["name", "address"])


def init_state(data):
    districts = sorted(data.complexes.district.unique())
    district = "송파구" if "송파구" in districts else districts[0]
    ids = data.complexes.loc[data.complexes.district == district, "complex_id"]
    present = set(data.quarterly.loc[data.quarterly.complex_id.isin(ids), "area_group"])
    areas = [area for area in AREA_LABELS if area in present]
    area = AREA_LABELS[2] if AREA_LABELS[2] in areas else areas[0]
    available = choices(data, district, area)
    preferred = available.loc[available.name == "리센츠", "complex_id"]
    cid = preferred.iloc[0] if len(preferred) else available.complex_id.iloc[0]
    draft = {"district": district, "area": area, "complex_id": cid}
    extras = [f"demo:{name}" for name in ["헬리오시티", "파크리오"]] if data.source == "demo" else []
    return {"draft": draft.copy(), "applied": draft.copy(), "extras": extras}


def ensure_widget(key, options, preferred):
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = preferred if preferred in options else options[0]


brand, mode_column = st.columns([3, 2], vertical_alignment="center")
with brand:
    html('<div class="brand"><span class="brand-mark">H</span><strong>우리집 가격체력</strong></div>')
with mode_column:
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "시연 화면" if st.query_params.get("mode") == "demo" else "실데이터"
    mode_label = st.radio("데이터 표시 모드", ["실데이터", "시연 화면"], key="view_mode", horizontal=True, label_visibility="collapsed")
mode = "demo" if mode_label == "시연 화면" else "real"
if st.query_params.get("mode") != mode:
    st.query_params["mode"] = mode

csv_path = Path(os.environ.get("APARTMENT_DATA_CSV", str(DEFAULT_CSV))).expanduser().resolve()
try:
    stamp = csv_path.stat() if mode == "real" else None
    signature = (mode, str(csv_path), stamp.st_mtime_ns if stamp else 0, stamp.st_size if stamp else 0, f"{PIPELINE_VERSION}:{DEMO_VERSION}")
    with st.spinner("아파트 거래 기록을 살펴보고 있어요…"):
        data = cached_dataset(*signature)
except (OSError, DataError) as exc:
    st.error(str(exc) if isinstance(exc, DataError) else "지정한 거래 CSV를 찾거나 읽을 수 없습니다.")
    st.info("CSV 위치와 필수 컬럼을 확인해 주세요. 위에서 시연 화면을 선택하면 예시 서비스를 볼 수 있어요.")
    st.stop()

state_key = f"domain_{mode}"
if state_key not in st.session_state or st.session_state[state_key].get("signature") != signature:
    old = st.session_state.get(state_key)
    fresh = init_state(data)
    # A replaced CSV must not leave stale widget or applied IDs in the page.
    if old:
        for field in ["district", "area", "complex", "extras"]:
            st.session_state.pop(f"{mode}_{field}", None)
    fresh["signature"] = signature
    st.session_state[state_key] = fresh
state = st.session_state[state_key]
draft = state["draft"]

if mode == "demo":
    html('<div class="source-strip demo"><span class="status-dot"></span><b>예시 데이터로 보는 시연 화면</b><span>2021년 3분기 ~ 2026년 2분기 · 가격·거래·점수·시설은 시연용이에요</span></div>')
else:
    html(f'<div class="source-strip"><span class="status-dot"></span><b>실거래 CSV 기준</b><span>{data.start:%Y.%m.%d} ~ {data.end:%Y.%m.%d} · {data.diagnostics["유효 거래"]:,}건의 거래 표본</span></div>')

with st.container(key="search"):
    html('<div class="search-copy"><h1>궁금한 아파트를 골라보세요</h1><p>지역과 면적을 고르면 과거에 얼마나 버티고 회복했는지 쉽게 보여드려요.</p></div>')
    c1, c2, c3, c4 = st.columns([1, 1.15, 1.4, .4], gap="small", vertical_alignment="bottom")
    with c1:
        districts = sorted(data.complexes.district.unique())
        ensure_widget(f"{mode}_district", districts, draft["district"])
        district = st.selectbox("자치구", districts, key=f"{mode}_district")
    with c2:
        ids = data.complexes.loc[data.complexes.district == district, "complex_id"]
        present = set(data.quarterly.loc[data.quarterly.complex_id.isin(ids), "area_group"])
        areas = [a for a in AREA_LABELS if a in present]
        ensure_widget(f"{mode}_area", areas, draft["area"])
        area = st.selectbox("전용면적", areas, key=f"{mode}_area")
    available = choices(data, district, area)
    option_ids = available.complex_id.tolist()
    label_map = {row.complex_id: f"{row['name']} · {row.dong}" for _, row in available.iterrows()}
    repeated = available["name"].duplicated(keep=False)
    for _, row in available.loc[repeated].iterrows():
        label_map[row.complex_id] = f"{row['name']} · {row.address}"
    with c3:
        ensure_widget(f"{mode}_complex", option_ids, draft["complex_id"])
        cid = st.selectbox("아파트", option_ids, format_func=label_map.get, key=f"{mode}_complex")
    draft.update(district=district, area=area, complex_id=cid)
    with c4:
        apply = st.button("결과 보기", type="primary", width="stretch")
    if apply:
        state["applied"] = draft.copy()
        state["extras"] = [i for i in state["extras"] if i in option_ids and i != cid][:2]
        st.session_state.pop(f"{mode}_extras", None)

applied = state["applied"]
snapshot = build_snapshot(data, applied["complex_id"], applied["area"])
if draft != applied:
    st.caption(f'현재 결과: {snapshot.info["name"]} · {applied["area"]}. 변경한 조건은 ‘결과 보기’를 누르면 적용돼요.')
render_summary(snapshot)
# 거래가 충분하면 기존 화면, 부족하면 확인된 것만 보여주는 대체 화면으로 통째로 갈아 끼운다.
if snapshot.metrics["mdd"].value is not None:
    render_price_overview(overview_values(data, snapshot))
else:
    render_sparse_overview(sparse_values(data, snapshot))
with st.container(key="chart-views"):
    trend_tab, distribution_tab = st.tabs(["가격 추이", "단지 분포"])
    with trend_tab:
        render_chart(snapshot)
    with distribution_tab:
        render_distribution(data, snapshot)

section("다른 후보와 비교해보세요", "선택한 아파트를 포함해 최대 3개 단지를 같은 조건으로 비교해요.", quarter_label(data.reference))
compare_candidates = choices(data, applied["district"], applied["area"])
extra_options = compare_candidates.loc[compare_candidates.complex_id != applied["complex_id"], "complex_id"].tolist()
compare_names = {r.complex_id: f"{r['name']} · {r.dong}" for _, r in compare_candidates.iterrows()}
for _, row in compare_candidates.loc[compare_candidates["name"].duplicated(keep=False)].iterrows():
    compare_names[row.complex_id] = f"{row['name']} · {row.address}"
extra_key = f"{mode}_extras"
if extra_key not in st.session_state:
    st.session_state[extra_key] = [i for i in state["extras"] if i in extra_options][:2]
else:
    cleaned = [i for i in st.session_state[extra_key] if i in extra_options][:2]
    if cleaned != st.session_state[extra_key]:
        st.session_state[extra_key] = cleaned
with st.container(key="comparison-controls"):
    fixed, selectable = st.columns([1, 3], vertical_alignment="center")
    with fixed:
        html(f'<div class="fixed-candidate"><small>기준 단지</small><span>{h(snapshot.info["name"])}</span></div>')
    with selectable:
        extras = st.multiselect("추가 비교 단지 (최대 2개)", extra_options, format_func=compare_names.get,
                                key=extra_key, max_selections=2, placeholder="함께 비교할 아파트를 골라주세요")
        state["extras"] = list(extras)
render_comparison([snapshot] + [build_snapshot(data, i, applied["area"]) for i in extras], applied["complex_id"])

section("살기 편한 동네인지도 볼까요?", "내 일상에 중요한 시설만 쉽게 확인하세요.", "생활권 양호 · 예시" if mode == "demo" else "직선거리 기준")
render_facilities(snapshot)
render_data_notes(data, snapshot)
