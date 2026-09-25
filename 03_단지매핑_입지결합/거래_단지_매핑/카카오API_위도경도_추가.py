"""카카오 주소 검색 API로 거래 CSV에 위도, 경도 컬럼을 추가합니다.

Python 3.9 이상 / 추가 패키지 설치 불필요.
실행 전 환경변수 KAKAO_REST_API_KEY에 REST API 키를 설정하세요.
(PowerShell 예시: $env:KAKAO_REST_API_KEY = "발급받은_키")
원본 CSV는 보존하며 결과, 조회기록, 재실행용 캐시를 거래_단지_DATA 폴더에 저장합니다.
좌표는 조회 시점의 주소 대표 위치이며 거래 당시 개별 동/호의 위치는 아닙니다.
공식 문서: https://developers.kakao.com/docs/ko/local/dev-guide#address-coord
"""

import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")

BASE_DIR = Path(__file__).resolve().parents[2] / "거래_단지_DATA"
INPUT_CSV = BASE_DIR / "아파트_거래_DATA_단지DATA_매핑_상위5000행.csv"
OUTPUT_CSV = INPUT_CSV.with_name(INPUT_CSV.stem + "_위도경도.csv")
LOG_CSV = INPUT_CSV.with_name(INPUT_CSV.stem + "_좌표조회기록.csv")
CACHE_JSON = BASE_DIR / "카카오_주소좌표_캐시_v1.json"
API_URL = "https://dapi.kakao.com/v2/local/search/address.json"
REQUEST_INTERVAL = 0.15
MAX_ATTEMPTS = 4


def clean(value):
    value = " ".join(str(value or "").split())
    return "" if value.lower() in {"", "-", "nan", "none", "null"} else value


def normalize_address(value):
    """시도 표기의 차이와 공백만 정규화해 엉뚱한 주소 선택을 방지합니다."""
    value = clean(value)
    aliases = {
        "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
        "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
        "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
        "강원특별자치도": "강원", "강원도": "강원",
        "충청북도": "충북", "충청남도": "충남",
        "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남",
        "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주",
    }
    parts = value.split()
    if parts:
        parts[0] = aliases.get(parts[0], parts[0])
    return "".join(parts)


def address_candidates(row):
    """거래 주소 우선, 매칭 완료된 단지의 주소를 후순위로 사용합니다."""
    candidates = [clean(row.get("도로명주소"))]
    district = clean(row.get("시군구"))
    lot = clean(row.get("번지"))
    if not lot:
        main, sub = clean(row.get("본번")), clean(row.get("부번"))
        if main.isdigit() and int(main) > 0 and (not sub or sub.isdigit()):
            lot = str(int(main))
            if sub and int(sub):
                lot += "-" + str(int(sub))
    if district and re.fullmatch(r"(?:산\s*)?\d+(?:-\d+)?", lot):
        candidates.append(f"{district} {lot}")
    if clean(row.get("단지매칭상태")) == "매칭완료":
        candidates.extend([clean(row.get("단지_도로명주소")),
                           clean(row.get("단지_법정동주소"))])
    return list(dict.fromkeys(value for value in candidates if value))


def request_address(query, api_key):
    params = urlencode({"query": query, "analyze_type": "exact", "size": 30})
    request = Request(API_URL + "?" + params,
                      headers={"Authorization": "KakaoAK " + api_key})
    for attempt in range(MAX_ATTEMPTS):
        time.sleep(REQUEST_INTERVAL)
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
                raise RuntimeError("카카오 API 응답 형식이 올바르지 않습니다.")
            return payload
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status in (401, 403):
                raise RuntimeError(
                    f"HTTP {status}: REST API 키와 카카오맵 API 사용 권한을 확인하세요."
                ) from None
            if status != 429 and not 500 <= status < 600:
                raise RuntimeError(f"카카오 API 요청 실패: HTTP {status}") from None
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(
                    f"HTTP {status}: 재시도 한도 초과. 호출 한도 또는 서버 상태를 확인 후 재실행하세요."
                ) from None
        except (URLError, TimeoutError, ConnectionError):
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError("네트워크 연결 실패. 연결 확인 후 재실행하세요.") from None
        time.sleep(2 ** (attempt + 1))


def select_coordinate(query, payload):
    documents = payload["documents"]
    matches = {}
    for doc in documents:
        if doc.get("address_type") not in {"REGION_ADDR", "ROAD_ADDR"}:
            continue  # 동/도로 자체의 중심 좌표는 사용하지 않습니다.
        names = [doc.get("address_name"),
                 (doc.get("address") or {}).get("address_name"),
                 (doc.get("road_address") or {}).get("address_name")]
        if normalize_address(query) not in {normalize_address(name) for name in names if name}:
            continue
        try:
            latitude, longitude = float(doc["y"]), float(doc["x"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(latitude) and math.isfinite(longitude)
                and -90 <= latitude <= 90 and -180 <= longitude <= 180):
            continue
        matches[(latitude, longitude)] = {
            "위도": str(doc["y"]), "경도": str(doc["x"]),
            "상태": "성공", "응답주소": doc.get("address_name", ""),
        }
    # 전체 결과가 한 페이지를 넘으면 보지 못한 후보가 있으므로 확정하지 않습니다.
    if len(matches) == 1 and payload.get("meta", {}).get("is_end") is True:
        return next(iter(matches.values()))
    reason = "검색결과없음" if not documents else "주소불일치_또는_복수후보"
    return {"위도": "", "경도": "", "상태": reason, "응답주소": ""}


def save_cache(cache, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(api_key, input_csv=INPUT_CSV, output_csv=OUTPUT_CSV,
        log_csv=LOG_CSV, cache_json=CACHE_JSON):
    if not api_key.strip():
        raise RuntimeError("환경변수 KAKAO_REST_API_KEY에 REST API 키를 설정하세요.")
    api_key = api_key.strip()
    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns or "도로명주소" not in columns:
            raise RuntimeError("입력 CSV에서 '도로명주소' 컬럼을 찾을 수 없습니다.")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise RuntimeError("입력 CSV에 열 개수가 맞지 않는 행이 있습니다.")
    if {"위도", "경도"}.intersection(columns):
        raise RuntimeError("입력 CSV에 이미 좌표 컬럼이 있습니다. 최초 원본 파일을 사용하세요.")

    cache = json.loads(cache_json.read_text(encoding="utf-8")) if cache_json.exists() else {}
    output_tmp = output_csv.with_suffix(".partial.csv")
    log_tmp = log_csv.with_suffix(".partial.csv")
    success = 0
    print(f"총 {len(rows):,}행 처리 시작 (기존 캐시 {len(cache):,}개)")
    try:
        with output_tmp.open("w", encoding="utf-8-sig", newline="") as out, \
                log_tmp.open("w", encoding="utf-8-sig", newline="") as log:
            writer = csv.DictWriter(out, fieldnames=columns + ["위도", "경도"])
            audit = csv.DictWriter(log, fieldnames=["데이터행번호", "NO", "단지명", "상태",
                                                   "사용주소", "응답주소", "시도기록"])
            writer.writeheader()
            audit.writeheader()
            for number, row in enumerate(rows, 1):
                result = {"위도": "", "경도": "", "상태": "주소없음", "응답주소": ""}
                attempts, selected = [], ""
                for query in address_candidates(row):
                    if query not in cache:
                        cache[query] = select_coordinate(query, request_address(query, api_key))
                        if len(cache) % 50 == 0:
                            save_cache(cache, cache_json)
                    result = cache[query]
                    attempts.append(f"{query}: {result['상태']}")
                    if result["상태"] == "성공":
                        selected = query
                        success += 1
                        break
                writer.writerow({**row, "위도": result["위도"], "경도": result["경도"]})
                audit.writerow({"데이터행번호": number, "NO": row.get("NO", ""),
                                "단지명": row.get("단지명", ""), "상태": result["상태"],
                                "사용주소": selected, "응답주소": result["응답주소"],
                                "시도기록": " | ".join(attempts)})
                if number % 100 == 0 or number == len(rows):
                    print(f"{number:,}/{len(rows):,}행 완료 | 성공 {success:,} | 미확정 {number-success:,}")
    finally:
        # 중단해도 완료된 주소 조회는 보관합니다. 재실행하면 캐시를 재사용합니다.
        save_cache(cache, cache_json)

    with output_tmp.open("r", encoding="utf-8-sig", newline="") as check:
        verified = csv.DictReader(check)
        if verified.fieldnames != columns + ["위도", "경도"]:
            raise RuntimeError("출력 컬럼 검증 실패")
        count = 0
        for count, saved in enumerate(verified, 1):
            if count > len(rows) or any(saved[c] != rows[count - 1][c] for c in columns):
                raise RuntimeError("기존 데이터 또는 행 순서 검증 실패")
        if count != len(rows):
            raise RuntimeError("출력 행 수 검증 실패")
    output_tmp.replace(output_csv)
    log_tmp.replace(log_csv)
    print(f"저장 완료: {output_csv}\n조회 기록: {log_csv}")
    print(f"성공 {success:,}행 / 좌표 빈칸 {len(rows)-success:,}행")


if __name__ == "__main__":
    try:
        run(KAKAO_REST_API_KEY)
    except KeyboardInterrupt:
        print("\n중단했습니다. 다시 실행하면 저장된 주소 캐시를 재사용합니다.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, ValueError, csv.Error) as error:
        print(f"오류: {error}\n실행 중 중단된 경우 .partial.csv는 미완성 파일입니다.", file=sys.stderr)
        sys.exit(1)
