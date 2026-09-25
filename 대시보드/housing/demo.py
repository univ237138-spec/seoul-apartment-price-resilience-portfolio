"""Deterministic presentation fixtures, never substituted into real data."""
import numpy as np
import pandas as pd

from .models import AREA_LABELS, Dataset, FACILITY_TYPES


DEMO_VERSION = "1.0"
# Fixed illustrative scores on the live 방어 20 / 회복 80 scale. The demo screen holds four
# complexes, too few for the peer ranking the real screen uses, so these stay hand-set.
SEEDS = [
    ("리센츠", "잠실동", 3540.0, 2389.0, 4118.0, "B+", [6, 58], [40, 40, 36, 89]),
    ("헬리오시티", "가락동", 3295 / 1.176, 3295 / 1.176 * .74, 3295.0, "A", [10, 62], [32, 38, 51, 65]),
    ("파크리오", "신천동", 3361 / 1.241, 3361 / 1.241 * .748, 3361.0, "A", [11, 68], [30, 33, 45, 54]),
    ("문정시영", "문정동", 1800.0, 1260.0, 1650.0, "C+", [7, 22], [8, 7, 9, 12]),
]


def demo_dataset() -> Dataset:
    quarters = pd.period_range("2021Q3", "2026Q2", freq="Q")
    rows, complexes, facilities, scores = [], [], [], {}
    for idx, (name, dong, peak, trough, current, grade, parts, counts) in enumerate(SEEDS):
        cid = f"demo:{name}"
        prices = np.interp(np.arange(len(quarters)), [0, 5, 10, 18, 19],
                           [peak, trough, trough + .38 * (current - trough), current * 1.02, current])
        for i, q in enumerate(quarters):
            rows.append({"complex_id": cid, "area_group": AREA_LABELS[2], "quarter": q,
                         "price": float(prices[i]), "count": counts[i - 16] if i >= 16 else 20 + i,
                         "complete": True})
        complexes.append({"complex_id": cid, "name": name, "district": "송파구", "dong": dong,
                          "address": f"서울특별시 송파구 {dong} · 시연용 예시",
                          "latitude": None, "longitude": None, "households": None,
                          "built_year": None, "parking_ground": None, "parking_underground": None,
                          "bundle_conflict": False})
        scores[(cid, AREA_LABELS[2])] = {"grade": grade, "total": sum(parts), "defense": parts[0],
                                         "recovery": parts[1], "rank": None, "peers": len(SEEDS)}
        for kind, distance, label in zip(FACILITY_TYPES, [480, 210, 350, 620],
                                         ["가까운 역", "가까운 학교", "가까운 공원", "장보기 시설"]):
            facilities.append({"complex_id": cid, "kind": kind, "name": label,
                               "subtype": "시연용 예시", "distance": distance + 30 * idx,
                               "latitude": None, "longitude": None, "status": "ok", "map_valid": False,
                               "count": (5 + idx if kind == "학교" else 4 + idx) if kind in ["학교", "대규모점포"] else None,
                               "count_radius": 1 if kind == "학교" else 2 if kind == "대규모점포" else None})
    return Dataset("demo", pd.DataFrame(rows), pd.DataFrame(complexes), pd.DataFrame(facilities),
                   quarters[0].start_time, quarters[-1].end_time.normalize(), quarters[-1], False,
                   {"데이터": "화면 검증용 고정 예시", "버전": DEMO_VERSION, "실제 거래 사용": "없음"}, scores)
