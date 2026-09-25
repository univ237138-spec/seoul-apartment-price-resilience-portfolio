import math
import textwrap

import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from pydeck.types import String

from .models import Snapshot, quarter_label
from .facility_icons import STYLES, icon_url, rgb


GREEN = "#167254"


def price_chart(s: Snapshot, period_range=None, zero_start=False) -> go.Figure:
    frame = s.series
    x = frame.quarter.astype(str).tolist()
    y = frame.price.to_numpy(dtype=float)
    counts = frame["count"].fillna(0).astype(int).tolist()
    labels = [quarter_label(q) for q in x]
    if period_range and period_range[0] in x and period_range[1] in x:
        view_start, view_end = x.index(period_range[0]), x.index(period_range[1])
    else:
        view_start, view_end = 0, len(x) - 1
    fig = go.Figure()
    notes = [" · ".join(note for note in [
        "부분 분기 · 잠정값" if s.partial and q == str(s.reference) else "",
        "거래 3건 미만 · 장기 지표에서 제외" if n < 3 else ""] if note) for q, n in zip(x, counts)]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", connectgaps=False,
        line={"color": GREEN, "width": 4}, marker={"size": 6, "color": GREEN,
            "symbol": ["circle-open" if n < 3 else "circle" for n in counts]},
        fill="tozeroy", fillcolor="rgba(22,114,84,.10)",
        customdata=list(zip(labels, counts, notes)),
        hovertemplate="%{customdata[0]}<br><b>%{y:,.1f}만원/㎡</b><br>거래 %{customdata[1]}건<br>%{customdata[2]}<extra></extra>",
        name="분기 중위가격",
    ))
    for key, label in [("peak", "고점"), ("trough", "저점"), ("price", "기준 가격")]:
        metric = s.metrics[key]
        if metric.value is not None:
            fig.add_trace(go.Scatter(x=[metric.period], y=[metric.value], mode="markers+text",
                                     marker={"size": 12, "color": GREEN}, text=[label],
                                     textposition="top center", textfont={"color": "#18372e", "size": 13},
                                     hoverinfo="skip", showlegend=False))
    visible = y[view_start:view_end + 1]
    valid = visible[np.isfinite(visible)]
    if len(valid):
        span = max(float(valid.max() - valid.min()), float(valid.max()) * .18)
        lower = max(0, float(valid.min()) - span * .20)
        upper = float(valid.max()) + span * .24
        y_range = [0 if zero_start else lower, upper]
    else:
        y_range = [0, 1]
    view_len = view_end - view_start + 1
    tick_indices = sorted(set([view_start, view_end] +
                              list(range(view_start, view_end + 1, max(1, view_len // 4)))))
    fig.update_layout(
        height=305, margin={"l": 8, "r": 20, "t": 24, "b": 15}, showlegend=False,
        font={"family": "Malgun Gothic, sans-serif", "size": 12, "color": "#74847f"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"type": "category", "categoryorder": "array", "categoryarray": x,
               "range": [x[view_start], x[view_end]],
               "tickmode": "array", "tickvals": [x[i] for i in tick_indices],
               "ticktext": [x[i].replace("Q", " Q") for i in tick_indices], "showgrid": False,
               "title": {"text": "거래 시기(분기)", "font": {"size": 11}}},
        yaxis={"range": y_range, "gridcolor": "#dfe7e2", "zeroline": False,
               "tickformat": ",.0f", "title": {"text": "만원/㎡", "font": {"size": 11}}},
        hoverlabel={"bgcolor": "#18372e", "font": {"color": "white"}},
    )
    if y_range[0] > 0:
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="bottom",
            text="// 0 생략", showarrow=False,
            font={"size": 10, "color": "#74847f"},
            bgcolor="rgba(255,254,250,.92)", borderpad=2,
        )
    return fig


def radius_polygon(latitude, longitude, meters=1000):
    """Geodesic ring, in lon/lat coordinates as required by deck.gl."""
    lat, lon = math.radians(latitude), math.radians(longitude)
    distance = meters / 6371008.8
    vertices = []
    for angle in np.linspace(0, 2 * math.pi, 73):
        new_lat = math.asin(math.sin(lat) * math.cos(distance) + math.cos(lat) * math.sin(distance) * math.cos(angle))
        new_lon = lon + math.atan2(math.sin(angle) * math.sin(distance) * math.cos(lat),
                                   math.cos(distance) - math.sin(lat) * math.sin(new_lat))
        vertices.append([math.degrees(new_lon), math.degrees(new_lat)])
    return vertices


def facility_map(s: Snapshot) -> pdk.Deck | None:
    lat, lon = s.info["latitude"], s.info["longitude"]
    if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
        return None
    pins = [{"position": [lon, lat], "name": s.info["name"], "detail": "선택한 아파트",
             "kind": "아파트"}]
    for _, row in s.facilities.iterrows():
        if row.map_valid:
            pins.append({"position": [row.longitude, row.latitude], "name": row["name"],
                         "detail": f'{STYLES[row.kind]["label"]} · 직선거리 {row.distance:,.0f}m', "kind": row.kind})
    for pin in pins:
        kind = pin["kind"]
        pin["color"] = rgb(kind)
        pin["icon"] = {"url": icon_url(kind), "width": 96, "height": 96, "mask": False}
        pin["size"] = 42 if kind == "아파트" else 38
        # Labels face away from the apartment; coordinates themselves never move.
        left = pin["position"][0] < lon
        pin["anchor"] = "end" if left else "start"
        pin["offset"] = [-25 if left else 27, -18 if kind == "아파트" else 8]
        if kind in ("지하철", "공원", "대규모점포"):
            pin["anchor"] = "middle"
            pin["offset"] = [0, 38]
        name = str(pin["name"])
        pin["label"] = "\n".join(textwrap.wrap(name, width=10))
    # pydeck string constants must use String; bare "pixels" becomes an accessor.
    # https://deckgl.readthedocs.io/en/latest/layer.html#passing-string-constants
    layers = [
        pdk.Layer("PolygonLayer", [{"polygon": radius_polygon(lat, lon)}], id="facility-radius", get_polygon="polygon",
                  stroked=True, filled=True, get_fill_color=[22, 114, 84, 15],
                  get_line_color=[22, 114, 84, 135], get_line_width=2, line_width_units=String("pixels")),
        pdk.Layer("IconLayer", pins, id="facility-pins", get_position="position", get_icon="icon",
                  get_size="size", size_units=String("pixels"), billboard=True, pickable=True),
        pdk.Layer("TextLayer", pins, id="facility-names", get_position="position", get_text="label",
                  get_color="color", get_size=16, size_units=String("pixels"),
                  get_text_anchor="anchor", get_alignment_baseline=String("center"), get_pixel_offset="offset",
                  font_family=String("Malgun Gothic, Apple SD Gothic Neo, sans-serif"), font_weight=700,
                  character_set=String("auto"), background=True, get_background_color=[255, 255, 255, 235],
                  background_padding=[5, 4], pickable=True),
    ]
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13.7),
                    map_provider="carto", map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    tooltip={"text": "{name}\n{detail}", "style": {"backgroundColor": "#18372e", "color": "white"}})
