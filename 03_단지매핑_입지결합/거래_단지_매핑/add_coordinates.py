"""
아파트 거래_단지 매핑 CSV에 위도/경도 컬럼을 추가하는 스크립트.

동작 방식
- 원본 CSV의 610,000+ 행을 전부 지오코딩하지 않고, 고유 도로명주소만 추출해 지오코딩한 뒤
  원본 행에 병합(merge)한다. (고유 도로명주소는 약 9천여 개 수준)
- 카카오 로컬 API(주소 검색)로 좌표를 조회한다. 도로명주소 검색이 실패하면
  지번주소(시군구+번지)로 재시도하고, 그래도 실패하면 단지명 키워드 검색으로 재시도한다.
- 조회 결과는 JSON 캐시 파일에 저장되어, 중간에 중단되어도 이어서 실행할 수 있다.

사전 준비
    PowerShell에서 카카오 REST API 키를 환경변수로 설정:
        $env:KAKAO_REST_API_KEY = "발급받은_REST_API_키"

실행
    python 03_단지매핑_입지결합/거래_단지_매핑/add_coordinates.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "거래_단지_DATA"
INPUT_CSV = DATA_DIR / "아파트_거래_DATA_단지DATA_매핑.csv"
OUTPUT_CSV = DATA_DIR / "아파트_거래_DATA_단지DATA_매핑_좌표추가.csv"
CACHE_JSON = DATA_DIR / "geocode_cache.json"

KAKAO_API_KEY = os.environ.get("KAKAO_REST_API_KEY")
ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

REQUEST_INTERVAL_SEC = 0.05  # 카카오 초당 호출 제한 대비 여유
SAVE_EVERY = 200  # 이 횟수마다 캐시 파일 저장


def load_cache():
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0)


def geocode_by_address(session, query):
    if not query or (isinstance(query, float) and pd.isna(query)):
        return None
    resp = session.get(
        ADDRESS_URL,
        params={"query": query},
        headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"},
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    docs = resp.json().get("documents", [])
    if not docs:
        return None
    doc = docs[0]
    return float(doc["y"]), float(doc["x"])  # 위도(y), 경도(x)


def geocode_by_keyword(session, query):
    if not query or (isinstance(query, float) and pd.isna(query)):
        return None
    resp = session.get(
        KEYWORD_URL,
        params={"query": query},
        headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"},
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    docs = resp.json().get("documents", [])
    if not docs:
        return None
    doc = docs[0]
    return float(doc["y"]), float(doc["x"])


def main():
    if not KAKAO_API_KEY:
        print("환경변수 KAKAO_REST_API_KEY 가 설정되어 있지 않습니다.")
        print('PowerShell 예시: $env:KAKAO_REST_API_KEY = "발급받은_키"')
        sys.exit(1)

    print("CSV 로딩 중...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
    print(f"전체 행 수: {len(df):,}")

    # 지오코딩용 고유 주소 목록 (도로명주소 기준), 지번/단지명 폴백 정보 함께 보관
    addr_df = (
        df[["도로명주소", "단지_법정동주소", "시군구", "번지", "단지명"]]
        .drop_duplicates(subset=["도로명주소"])
        .reset_index(drop=True)
    )
    print(f"고유 도로명주소 수: {len(addr_df):,}")

    cache = load_cache()
    session = requests.Session()

    todo = addr_df[~addr_df["도로명주소"].isin(cache.keys())]
    print(f"신규 지오코딩 대상: {len(todo):,} (캐시에 이미 있는 항목 제외)")

    processed = 0
    for _, row in todo.iterrows():
        road_addr = row["도로명주소"]
        jibun_addr = row["단지_법정동주소"]
        if pd.isna(jibun_addr):
            si_gun_gu = row["시군구"] if not pd.isna(row["시군구"]) else ""
            bunji = row["번지"] if not pd.isna(row["번지"]) else ""
            jibun_addr = f"{si_gun_gu} {bunji}".strip()
        keyword = row["단지명"]
        if not pd.isna(row["시군구"]):
            keyword = f"{row['시군구']} {keyword}"

        result = geocode_by_address(session, road_addr)
        source = "road"
        if result is None:
            result = geocode_by_address(session, jibun_addr)
            source = "jibun"
        if result is None:
            result = geocode_by_keyword(session, keyword)
            source = "keyword"

        if result is not None:
            lat, lon = result
            cache[road_addr] = {"lat": lat, "lon": lon, "source": source}
        else:
            cache[road_addr] = {"lat": None, "lon": None, "source": "failed"}

        processed += 1
        if processed % SAVE_EVERY == 0:
            save_cache(cache)
            print(f"  진행: {processed:,}/{len(todo):,} (캐시 저장됨)")

        time.sleep(REQUEST_INTERVAL_SEC)

    save_cache(cache)

    n_success = sum(1 for v in cache.values() if v["lat"] is not None)
    n_fail = sum(1 for v in cache.values() if v["lat"] is None)
    print(f"지오코딩 완료: 성공 {n_success:,} / 실패 {n_fail:,}")

    # 원본 행에 위도/경도 병합
    lat_map = {k: v["lat"] for k, v in cache.items()}
    lon_map = {k: v["lon"] for k, v in cache.items()}
    df["위도"] = df["도로명주소"].map(lat_map)
    df["경도"] = df["도로명주소"].map(lon_map)

    missing = df["위도"].isna().sum()
    print(f"좌표 미부여 행: {missing:,} / {len(df):,}")

    df.to_csv(OUTPUT_CSV, encoding="utf-8-sig", index=False)
    print(f"저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
