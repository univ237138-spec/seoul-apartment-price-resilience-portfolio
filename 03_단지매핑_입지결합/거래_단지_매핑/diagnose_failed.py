"""
지오코딩 실패 10건의 원인을 진단하는 스크립트.
add_coordinates.py와 동일한 방식(도로명주소 -> 지번주소 -> 키워드)으로
각 단계별 카카오 API 원본 응답(status, 오류 메시지)을 출력한다.

실행 전: $env:KAKAO_REST_API_KEY = "발급받은_키"  (본인 PowerShell에서만 설정)
실행:    python diagnose_failed.py
"""

import json
import os
import sys

import requests

KAKAO_API_KEY = os.environ.get("KAKAO_REST_API_KEY")
ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

CASES = [
    ("서울특별시 송파구 오금로 557", "서울특별시 송파구 거여동 34", "서울특별시 송파구 메이저A동"),
    ("서울특별시 강동구 고덕로80길 13", "서울특별시 강동구 상일동 121", "서울특별시 강동구 주공3"),
    ("서울특별시 강동구 상암로79길 62", "서울특별시 강동구 상일동 187", "서울특별시 강동구 주공7"),
    ("서울특별시 구로구 구일로 18", "서울특별시 구로구 구로동 685-280", "서울특별시 구로구 칠성3"),
    ("서울특별시 성북구 삼양로6길 31", "서울특별시 성북구 길음동 998", "서울특별시 성북구 프리젠트빌"),
    ("서울특별시 은평구 증산로3길 19-6", "서울특별시 은평구 증산동 214-13", "서울특별시 은평구 에쉐르"),
    ("서울특별시 강남구 언주로 406", "서울특별시 강남구 역삼동 712", "서울특별시 강남구 개나리4"),
    ("서울특별시 송파구 오금로61길 24", "서울특별시 송파구 거여동 199-2", "서울특별시 송파구 보화1차"),
    ("서울특별시 성북구 동소문로43길 47", "서울특별시 성북구 길음동 973-2", "서울특별시 성북구 두이노블레스(2)"),
    ("서울특별시 서초구 신반포로23길 31", "서울특별시 서초구 잠원동 74", "서울특별시 서초구 신반포14"),
]


def call(url, query, session):
    resp = session.get(
        url,
        params={"query": query},
        headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"},
        timeout=10,
    )
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    n_docs = len(body.get("documents", [])) if isinstance(body, dict) else "N/A"
    err = body.get("errorType") if isinstance(body, dict) else None
    msg = body.get("message") if isinstance(body, dict) else None
    return resp.status_code, n_docs, err, msg


def main():
    if not KAKAO_API_KEY:
        print("환경변수 KAKAO_REST_API_KEY 가 설정되어 있지 않습니다.")
        sys.exit(1)

    session = requests.Session()
    for road, jibun, keyword in CASES:
        print("=" * 70)
        print(f"도로명: {road}")
        sc, nd, err, msg = call(ADDRESS_URL, road, session)
        print(f"  [주소검색-도로명] status={sc} documents={nd} error={err} msg={msg}")

        print(f"지번:   {jibun}")
        sc, nd, err, msg = call(ADDRESS_URL, jibun, session)
        print(f"  [주소검색-지번]   status={sc} documents={nd} error={err} msg={msg}")

        print(f"키워드: {keyword}")
        sc, nd, err, msg = call(KEYWORD_URL, keyword, session)
        print(f"  [키워드검색]      status={sc} documents={nd} error={err} msg={msg}")

    print("=" * 70)
    print("완료. 위 출력 전체를 복사해서 알려주세요 (키는 출력되지 않습니다).")


if __name__ == "__main__":
    main()
