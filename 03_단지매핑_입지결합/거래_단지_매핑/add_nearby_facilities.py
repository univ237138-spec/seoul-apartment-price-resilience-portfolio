"""
아파트 거래 데이터(위도/경도 포함)에, 아파트_입지_DATA.csv의 시설(공원/지하철/대규모점포/학교)
중 직선거리 1km 이내에 있는 시설들을 컬럼으로 추가한다.

- 시설 유형별로 컬럼 2개씩 생성: "근처_XXX" (이름들, 가까운 순 쉼표 구분), "근처_XXX_거리m" (거리들, 같은 순서)
- 학교는 세부유형(초등학교/중학교/고등학교)별로 별도 컬럼 생성
- 계산은 (위도,경도) 고유 조합 단위로만 수행한 뒤 원본 행에 병합 (전체 행 아닌 고유 좌표 수만큼만 계산 -> 훨씬 빠름)

실행: python 03_단지매핑_입지결합/거래_단지_매핑/add_nearby_facilities.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRANSACTION_CSV = ROOT / "거래_단지_DATA" / "아파트_거래_DATA_단지DATA_매핑_좌표추가.csv"
FACILITY_CSV = ROOT / "01_원천데이터" / "아파트_입지_DATA.csv"
OUTPUT_CSV = ROOT / "거래_단지_DATA" / "아파트_거래_DATA_단지DATA_매핑_좌표추가_입지반영.csv"

RADIUS_M = 1000.0
EARTH_RADIUS_M = 6371000.0

# (컬럼 접두어, 시설유형, 세부유형 or None)
FACILITY_GROUPS = [
    ("공원", "공원", None),
    ("지하철역", "지하철", None),
    ("대규모점포", "대규모점포", None),
    ("초등학교", "학교", "초등학교"),
    ("중학교", "학교", "중학교"),
    ("고등학교", "학교", "고등학교"),
]


def haversine_matrix(lat1, lon1, lat2, lon2):
    """lat1/lon1: (N,) 배열, lat2/lon2: (M,) 배열 -> (N,M) 거리행렬(m)"""
    lat1r = np.radians(lat1)[:, None]
    lon1r = np.radians(lon1)[:, None]
    lat2r = np.radians(lat2)[None, :]
    lon2r = np.radians(lon2)[None, :]

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_M * c


def main():
    print("데이터 로딩 중...")
    df = pd.read_csv(TRANSACTION_CSV, encoding="utf-8-sig", low_memory=False)
    fac = pd.read_csv(FACILITY_CSV, encoding="utf-8-sig")

    print(f"거래 데이터 행 수: {len(df):,}")
    print(f"시설 데이터 행 수: {len(fac):,}")

    # 좌표 있는 고유 (위도, 경도) 조합만 추출
    coords = df[["위도", "경도"]].dropna().drop_duplicates().reset_index(drop=True)
    print(f"고유 좌표 조합 수: {len(coords):,}")

    lat_arr = coords["위도"].to_numpy()
    lon_arr = coords["경도"].to_numpy()

    result_cols = {prefix: [""] * len(coords) for prefix, _, _ in FACILITY_GROUPS}
    result_dist_cols = {prefix: [""] * len(coords) for prefix, _, _ in FACILITY_GROUPS}

    for prefix, ftype, subtype in FACILITY_GROUPS:
        if subtype is None:
            sub = fac[fac["시설유형"] == ftype]
        else:
            sub = fac[(fac["시설유형"] == ftype) & (fac["세부유형"] == subtype)]

        print(f"[{prefix}] 대상 시설 수: {len(sub):,}")
        if len(sub) == 0:
            continue

        fac_lat = sub["위도"].to_numpy()
        fac_lon = sub["경도"].to_numpy()
        fac_name = sub["시설명"].to_numpy()

        dist = haversine_matrix(lat_arr, lon_arr, fac_lat, fac_lon)  # (N, M)

        for i in range(len(coords)):
            row_dist = dist[i]
            within = np.where(row_dist <= RADIUS_M)[0]
            if len(within) == 0:
                continue
            order = within[np.argsort(row_dist[within])]
            names = fac_name[order]
            dists = row_dist[order]
            result_cols[prefix][i] = ", ".join(names)
            result_dist_cols[prefix][i] = ", ".join(str(round(d)) for d in dists)

    for prefix, _, _ in FACILITY_GROUPS:
        coords[f"근처_{prefix}"] = result_cols[prefix]
        coords[f"근처_{prefix}_거리m"] = result_dist_cols[prefix]

    print("원본 데이터에 병합 중...")
    df = df.merge(coords, on=["위도", "경도"], how="left")

    df.to_csv(OUTPUT_CSV, encoding="utf-8-sig", index=False)
    print(f"저장 완료: {OUTPUT_CSV}")

    for prefix, _, _ in FACILITY_GROUPS:
        has_any = (df[f"근처_{prefix}"].fillna("") != "").sum()
        print(f"  {prefix}: 1km 이내 시설 있는 행 {has_any:,} / {len(df):,}")


if __name__ == "__main__":
    main()
