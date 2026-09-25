"""
'_입지반영.csv'(전체 610,341행, 시설별 1km 이내 전체 목록 컬럼)를
스트림릿 대시보드(housing/data.py)가 기대하는 기존 스키마로 변환한다.

대시보드가 기대하는 시설 컬럼(FACILITY_TYPES = 지하철/학교/공원/대규모점포, 각 유형당
최근접 시설 1개): {kind}_시설명, {kind}_세부유형, {kind}_주소, {kind}_위도, {kind}_경도,
{kind}_거리_m, {kind}_입지원본행번호

- 학교는 초/중/고 구분 없이 '학교' 하나로 통합하고, 그중 가장 가까운 학교만 사용한다.
- 계산은 고유 (위도,경도) 조합 단위로만 수행한 뒤 원본 행에 병합한다.

실행: python 03_단지매핑_입지결합/거래_단지_매핑/convert_to_dashboard_schema.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "거래_단지_DATA" / "아파트_거래_DATA_단지DATA_매핑_좌표추가_입지반영.csv"
FACILITY_CSV = ROOT / "01_원천데이터" / "아파트_입지_DATA.csv"
OUTPUT_CSV = ROOT / "거래_단지_DATA" / "아파트_거래_DATA_대시보드_실데이터.csv"

EARTH_RADIUS_M = 6371000.0

# 대시보드 컬럼 접두어 -> 아파트_입지_DATA.csv의 시설유형
KIND_TO_FACILITY_TYPE = {
    "지하철": "지하철",
    "학교": "학교",       # 초/중/고 통합, 최근접 1개만 사용
    "공원": "공원",
    "대규모점포": "대규모점포",
}


def haversine_matrix(lat1, lon1, lat2, lon2):
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
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    fac = pd.read_csv(FACILITY_CSV, encoding="utf-8-sig").reset_index().rename(columns={"index": "_원본행번호"})

    print(f"거래 데이터 행 수: {len(df):,}")
    print(f"시설 데이터 행 수: {len(fac):,}")

    coords = df[["위도", "경도"]].dropna().drop_duplicates().reset_index(drop=True)
    print(f"고유 좌표 조합 수: {len(coords):,}")

    lat_arr = coords["위도"].to_numpy(dtype=float)
    lon_arr = coords["경도"].to_numpy(dtype=float)

    for prefix, ftype in KIND_TO_FACILITY_TYPE.items():
        sub = fac[fac["시설유형"] == ftype].reset_index(drop=True)
        print(f"[{prefix}] 대상 시설 수: {len(sub):,}")

        name_col = np.full(len(coords), "", dtype=object)
        subtype_col = np.full(len(coords), "", dtype=object)
        addr_col = np.full(len(coords), "", dtype=object)
        lat_col = np.full(len(coords), np.nan)
        lon_col = np.full(len(coords), np.nan)
        dist_col = np.full(len(coords), np.nan)
        origin_row_col = np.full(len(coords), "", dtype=object)

        if len(sub) > 0:
            fac_lat = sub["위도"].to_numpy(dtype=float)
            fac_lon = sub["경도"].to_numpy(dtype=float)
            dist = haversine_matrix(lat_arr, lon_arr, fac_lat, fac_lon)  # (N, M)
            nearest_idx = np.argmin(dist, axis=1)
            nearest_dist = dist[np.arange(len(coords)), nearest_idx]

            name_col = sub["시설명"].to_numpy()[nearest_idx]
            subtype_col = sub["세부유형"].fillna("").to_numpy()[nearest_idx]
            addr_col = sub["주소"].fillna("").to_numpy()[nearest_idx]
            lat_col = fac_lat[nearest_idx]
            lon_col = fac_lon[nearest_idx]
            dist_col = nearest_dist
            origin_row_col = sub["_원본행번호"].to_numpy()[nearest_idx]

        coords[f"{prefix}_시설명"] = name_col
        coords[f"{prefix}_세부유형"] = subtype_col
        coords[f"{prefix}_주소"] = addr_col
        coords[f"{prefix}_위도"] = lat_col
        coords[f"{prefix}_경도"] = lon_col
        coords[f"{prefix}_거리_m"] = dist_col
        coords[f"{prefix}_입지원본행번호"] = origin_row_col

    print("원본 데이터에 병합 중...")
    df = df.merge(coords, on=["위도", "경도"], how="left")

    df.to_csv(OUTPUT_CSV, encoding="utf-8-sig", index=False)
    print(f"저장 완료: {OUTPUT_CSV}")
    print(f"전체 컬럼 수: {len(df.columns)}")

    for prefix in KIND_TO_FACILITY_TYPE:
        within = (df[f"{prefix}_거리_m"] <= 1000).sum()
        print(f"  {prefix}: 1km 이내 매칭 행 {within:,} / {len(df):,}")


if __name__ == "__main__":
    main()
