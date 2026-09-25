"""Shared, font-independent pictograms for cards and both maps."""
from functools import lru_cache
from urllib.parse import quote


STYLES = {
    "아파트": {"label": "기준 아파트", "color": "#18372e"},
    "지하철": {"label": "지하철", "color": "#2475a0"},
    "학교": {"label": "학교", "color": "#a86413"},
    "공원": {"label": "공원", "color": "#21804a"},
    "대규모점포": {"label": "장보기", "color": "#8853a3"},
}

# Vector shapes stay identical on Windows, macOS and the WebGL map.
SHAPES = {
    "아파트": '<path d="M17 48V17h20v31M37 27h10v21"/><path d="M23 23h3m5 0h1m-9 7h3m5 0h1m-9 7h3m5 0h1m11-3h1m-1 7h1" stroke="white"/><path d="M25 48v-7h5v7" fill="white"/>',
    "지하철": '<rect x="19" y="14" width="26" height="33" rx="8"/><path d="M25 48l-4 5m18-5 4 5" fill="none"/><rect x="23" y="23" width="18" height="12" rx="2" fill="white" stroke="none"/><path d="M28 19h8" stroke="white"/><circle cx="25" cy="41" r="2" fill="white" stroke="none"/><circle cx="39" cy="41" r="2" fill="white" stroke="none"/>',
    "학교": '<path d="M13 48V29h38v19M23 48V25l9-8 9 8v23"/><path d="M32 17V10h8l-2 3h-6" fill="none"/><circle cx="32" cy="29" r="5" fill="white" stroke="none"/><path d="M32 26v3h3" stroke-width="1.5" fill="none"/><path d="M29 48V38h6v10" fill="white"/><path d="M17 35h2m-2 7h2m26-7h2m-2 7h2" stroke="white"/>',
    "공원": '<path d="M30 36h4v16h-4z"/><path d="M22 40c-13 0-14-15-5-19-2-14 19-17 23-6 13-1 18 19 6 24-3 2-8 2-11-1-4 4-9 4-13 2z"/><path d="M32 44V25m0 11-8-7m8 3 8-7" stroke="white" fill="none"/>',
    "대규모점포": '<path d="M16 28h32v22H16z"/><path d="M13 28l5-13h28l5 13z"/><path d="M22 16l-2 12m12-12v12m10-12 2 12" stroke="white"/><path d="M13 28q4 9 9 0 5 9 10 0 5 9 10 0 5 9 9 0" fill="white"/><rect x="21" y="36" width="10" height="8" rx="1" fill="white" stroke="none"/><path d="M36 50V36h7v14" fill="white"/>',
}


def rgb(kind):
    color = STYLES[kind]["color"].lstrip("#")
    return [int(color[i:i + 2], 16) for i in (0, 2, 4)]


@lru_cache(maxsize=5)
def icon_url(kind):
    color = STYLES[kind]["color"]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 64 64">'
           f'<rect x="2" y="2" width="60" height="60" rx="17" fill="white" stroke="{color}" stroke-width="2"/>'
           f'<g fill="{color}" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
           f'{SHAPES[kind]}</g></svg>')
    return "data:image/svg+xml;charset=utf-8," + quote(svg, safe="")


def icon_html(kind):
    return f'<img class="facility-symbol" src="{icon_url(kind)}" alt="{STYLES[kind]["label"]} 아이콘" width="40" height="40">'
