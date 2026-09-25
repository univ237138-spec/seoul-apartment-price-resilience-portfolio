"""
대시보드가 실제로 읽는 CSV(gzip 압축본)를 원본에서 다시 만든다.

- 대상 기간: 필터링 없음. 원본에 있는 전체 기간을 그대로 옮긴다.
- 컬럼: housing/data.py가 실제로 사용하는 컬럼만 유지 (REQUIRED + INFO_COLUMNS + FACILITY_COLUMNS)
- gzip 압축 저장 (pandas가 .gz 확장자를 자동 인식하므로 앱 코드 변경 불필요)

원본 아파트_거래_DATA_대시보드_실데이터.csv(842MB)는 로컬에만 있고 git에 커밋되지 않는다
(용량 제한). 이 스크립트가 만드는 .csv.gz가 로컬·테스트·Streamlit Cloud가 공통으로 쓰는
유일한 데이터 파일(housing.data.DEFAULT_CSV)이므로, 원본을 갱신했으면 이 스크립트를 다시
실행해서 커밋해야 배포본에도 반영된다.

용량 참고: 전체 610,341행 기준 약 95MB로, GitHub 파일당 100MB 제한에 여유가 크지 않다.
원본 거래가 크게 늘어나면(대략 640,000행 이상) 다시 이 한계에 걸릴 수 있으니, 그때는
Parquet 등 더 압축률이 높은 포맷으로 바꾸는 걸 검토한다.

실행: python 03_단지매핑_입지결합/거래_단지_매핑/build_cloud_dataset.py
"""

from pathlib import Path

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "대시보드"))
from housing.data import INFO_COLUMNS, REQUIRED
from housing.models import FACILITY_TYPES

INPUT_CSV = ROOT / "거래_단지_DATA" / "아파트_거래_DATA_대시보드_실데이터.csv"
OUTPUT_CSV_GZ = ROOT / "거래_단지_DATA" / "아파트_거래_DATA_대시보드_실데이터.csv.gz"

NEED_COLUMNS = sorted(set(
    REQUIRED + INFO_COLUMNS + ["해제사유발생일"]
    + [f"{kind}_{suffix}" for kind in FACILITY_TYPES
       for suffix in ["시설명", "세부유형", "주소", "위도", "경도", "거리_m", "입지원본행번호"]]
))


def main():
    print("데이터 로딩 중...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig",
                      usecols=lambda c: c in NEED_COLUMNS, dtype=str, low_memory=False)
    print(f"원본(컬럼 축소 후) 행/열: {df.shape}")
    print(f"계약년월 범위: {df['계약년월'].min()} ~ {df['계약년월'].max()}")

    df.to_csv(OUTPUT_CSV_GZ, index=False, encoding="utf-8-sig", compression="gzip")
    size_mb = Path(OUTPUT_CSV_GZ).stat().st_size / (1024 * 1024)
    print(f"저장 완료: {OUTPUT_CSV_GZ} ({size_mb:.1f} MB)")
    if size_mb > 90:
        print("경고: GitHub 파일당 100MB 제한에 근접했습니다. 다음 갱신 전에 포맷 교체를 검토하세요.")


if __name__ == "__main__":
    main()
