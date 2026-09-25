"""분석 보고서의 권장 타입을 적용해 새 CSV와 검증 보고서를 생성한다.

CSV는 타입 메타데이터를 저장하지 않으므로, 출력값을 타입에 맞는 표준 형식으로
정규화하고 ``load_typed_csv`` 함수에서 pandas 타입을 명시적으로 복원한다.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_CSV = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합.csv"
OUTPUT_CSV = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용.csv"
REPORT_MD = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_분석결과.md"
TEMP_CSV = BASE_DIR / ".아파트_실거래가_권장타입적용.tmp.csv"
CHUNK_SIZE = 100_000

EXPECTED_COLUMNS = [
    "NO",
    "시군구",
    "번지",
    "본번",
    "부번",
    "단지명",
    "전용면적(㎡)",
    "계약년월",
    "계약일",
    "거래금액(만원)",
    "동",
    "층",
    "매수자",
    "매도자",
    "건축년도",
    "도로명",
    "해제사유발생일",
    "거래유형",
    "중개사소재지",
    "등기일자",
]

STRING_COLUMNS = ["시군구", "번지", "본번", "부번", "단지명", "동", "도로명", "중개사소재지"]
INTEGER_COLUMNS = ["NO", "계약일", "층", "건축년도"]
CATEGORY_COLUMNS = ["매수자", "매도자", "거래유형"]
DATE_FORMATS = {"해제사유발생일": "%Y%m%d", "등기일자": "%y.%m.%d"}

TYPE_DETAILS = [
    ("NO", "정수형 식별자(Int64) 또는 문자열", "Int64", "정수", "숫자 식별자이므로 Int64 선택"),
    ("시군구", "문자열(string)", "string", "문자열", "원문 유지"),
    ("번지", "문자열(string)", "string", "문자열", "원문 유지"),
    ("본번", "문자열(string, 선행 0 보존)", "string", "4자리 문자열", "선행 0 보존"),
    ("부번", "문자열(string, 선행 0 보존)", "string", "4자리 문자열", "선행 0 보존"),
    ("단지명", "문자열(string)", "string", "문자열", "원문 유지"),
    ("전용면적(㎡)", "실수형(float64)", "float64", "소수", "숫자 변환"),
    ("계약년월", "연월형(YYYYMM; 문자열 또는 Period[M])", "period[M]", "YYYY-MM", "월 단위 Period로 변환"),
    ("계약일", "정수형(Int64)", "Int64", "정수", "숫자 변환"),
    ("거래금액(만원)", "정수형(Int64)", "Int64", "쉼표 없는 정수", "천 단위 쉼표 제거 후 숫자 변환"),
    ("동", "문자열(string)", "string", "문자열", "원문 유지"),
    ("층", "정수형(Int64)", "Int64", "정수", "숫자 변환"),
    ("매수자", "범주/문자열(category/string)", "category", "범주 문자열", "저카디널리티 범주형 선택"),
    ("매도자", "범주/문자열(category/string)", "category", "범주 문자열", "저카디널리티 범주형 선택"),
    ("건축년도", "정수형(Int64)", "Int64", "정수", "숫자 변환"),
    ("도로명", "문자열(string)", "string", "문자열", "원문 유지"),
    ("해제사유발생일", "날짜형(datetime64[ns])", "datetime64[ns]", "YYYY-MM-DD 또는 빈 값", "'-'를 결측으로 바꾸고 날짜 변환"),
    ("거래유형", "범주/문자열(category/string)", "category", "범주 문자열", "저카디널리티 범주형 선택"),
    ("중개사소재지", "문자열(string)", "string", "문자열", "원문 유지"),
    ("등기일자", "날짜형(datetime64[ns])", "datetime64[ns]", "YYYY-MM-DD 또는 빈 값", "'-'를 결측으로 바꾸고 날짜 변환"),
]


def sha256_file(path: Path) -> str:
    """파일의 SHA-256 해시를 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def human_bytes(size: int) -> str:
    """파일 크기를 읽기 쉬운 이진 단위로 표시한다."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.2f} {unit}"
        value /= 1024
    raise AssertionError("도달할 수 없는 코드")


def require_columns(columns: list[str]) -> None:
    """입력 스키마가 예상과 정확히 일치하는지 확인한다."""
    if columns != EXPECTED_COLUMNS:
        raise ValueError(
            "컬럼 구성이 예상과 다릅니다.\n"
            f"예상: {EXPECTED_COLUMNS}\n"
            f"실제: {columns}"
        )


def parse_date(series: pd.Series, fmt: str, column: str) -> pd.Series:
    """'-'와 빈 문자열을 결측값으로 바꾸고 지정 형식의 날짜로 변환한다."""
    cleaned = series.mask(series.isin(["-", ""]))
    parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce")
    invalid = cleaned.notna() & parsed.isna()
    if invalid.any():
        examples = cleaned[invalid].drop_duplicates().head(10).tolist()
        raise ValueError(f"{column}에 변환할 수 없는 날짜가 있습니다: {examples}")
    return parsed


def parse_numeric(series: pd.Series, column: str, *, integer: bool) -> pd.Series:
    """숫자 변환 실패를 허용하지 않고 정수 또는 실수로 변환한다."""
    cleaned = series.mask(series.eq(""))
    parsed = pd.to_numeric(cleaned, errors="coerce")
    invalid = cleaned.notna() & parsed.isna()
    if invalid.any():
        examples = cleaned[invalid].drop_duplicates().head(10).tolist()
        raise ValueError(f"{column}에 변환할 수 없는 숫자가 있습니다: {examples}")
    return parsed.astype("Int64" if integer else "float64")


def transform_source_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    """원본 형식의 데이터 조각에 보고서의 권장 타입을 적용한다."""
    require_columns(raw.columns.tolist())
    converted = raw.copy()

    for column in STRING_COLUMNS:
        converted[column] = converted[column].astype("string")

    for column in INTEGER_COLUMNS:
        converted[column] = parse_numeric(converted[column], column, integer=True)

    converted["전용면적(㎡)"] = parse_numeric(converted["전용면적(㎡)"], "전용면적(㎡)", integer=False)
    converted["거래금액(만원)"] = parse_numeric(
        converted["거래금액(만원)"].str.replace(",", "", regex=False),
        "거래금액(만원)",
        integer=True,
    )

    year_month = pd.to_datetime(converted["계약년월"], format="%Y%m", errors="coerce")
    invalid_year_month = converted["계약년월"].ne("") & year_month.isna()
    if invalid_year_month.any():
        examples = converted.loc[invalid_year_month, "계약년월"].drop_duplicates().head(10).tolist()
        raise ValueError(f"계약년월에 변환할 수 없는 값이 있습니다: {examples}")
    converted["계약년월"] = year_month.dt.to_period("M")

    for column, fmt in DATE_FORMATS.items():
        converted[column] = parse_date(converted[column], fmt, column)

    for column in CATEGORY_COLUMNS:
        converted[column] = converted[column].astype("category")

    return converted[EXPECTED_COLUMNS]


def load_typed_csv(path: Path | str = OUTPUT_CSV) -> pd.DataFrame:
    """새 CSV를 보고서 권장 pandas 타입으로 복원해 읽는다."""
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    require_columns(frame.columns.tolist())

    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in INTEGER_COLUMNS + ["거래금액(만원)"]:
        frame[column] = parse_numeric(frame[column], column, integer=True)
    frame["전용면적(㎡)"] = parse_numeric(frame["전용면적(㎡)"], "전용면적(㎡)", integer=False)

    year_month = pd.to_datetime(frame["계약년월"], format="%Y-%m", errors="coerce")
    if (frame["계약년월"].notna() & year_month.isna()).any():
        raise ValueError("출력 CSV의 계약년월 형식이 YYYY-MM이 아닙니다.")
    frame["계약년월"] = year_month.dt.to_period("M")

    for column in DATE_FORMATS:
        parsed = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="coerce")
        if (frame[column].notna() & parsed.isna()).any():
            raise ValueError(f"출력 CSV의 {column} 형식이 YYYY-MM-DD가 아닙니다.")
        frame[column] = parsed

    for column in CATEGORY_COLUMNS:
        frame[column] = frame[column].astype("category")
    return frame[EXPECTED_COLUMNS]


def validate_serialized_chunk(frame: pd.DataFrame) -> None:
    """CSV에 기록된 텍스트 표현이 약속한 스키마를 만족하는지 검증한다."""
    require_columns(frame.columns.tolist())

    regex_rules = {
        "NO": r"^-?\d+$",
        "본번": r"^\d{4}$",
        "부번": r"^\d{4}$",
        "계약년월": r"^\d{4}-\d{2}$",
        "계약일": r"^\d+$",
        "거래금액(만원)": r"^-?\d+$",
        "층": r"^-?\d+$",
        "건축년도": r"^\d+$",
    }
    for column, pattern in regex_rules.items():
        invalid = frame[column].notna() & ~frame[column].str.fullmatch(pattern)
        if invalid.any():
            examples = frame.loc[invalid, column].drop_duplicates().head(10).tolist()
            raise ValueError(f"출력 {column} 형식 검증 실패: {examples}")

    numeric_area = pd.to_numeric(frame["전용면적(㎡)"], errors="coerce")
    if (frame["전용면적(㎡)"].notna() & numeric_area.isna()).any():
        raise ValueError("출력 전용면적(㎡) 숫자 형식 검증 실패")

    for column in DATE_FORMATS:
        parsed = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="coerce")
        if (frame[column].notna() & parsed.isna()).any():
            raise ValueError(f"출력 {column} 날짜 형식 검증 실패")


def write_report(
    *,
    source_hash: str,
    row_count: int,
    null_counts: Counter[str],
    dtype_counts: Counter[str],
    output_hash: str,
) -> None:
    """변환 결과와 검증 내역을 Markdown 보고서로 저장한다."""
    source_size = SOURCE_CSV.stat().st_size
    output_size = OUTPUT_CSV.stat().st_size
    lines = [
        "# 아파트(매매) 실거래가 권장 타입 적용 결과",
        "",
        f"- 작업 시각: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 원본 파일: `{SOURCE_CSV.name}`",
        f"- 새 파일: `{OUTPUT_CSV.name}`",
        f"- 데이터 크기: **{row_count:,}행 × {len(EXPECTED_COLUMNS):,}열**",
        f"- 원본 크기: **{source_size:,} bytes ({human_bytes(source_size)})**",
        f"- 새 CSV 크기: **{output_size:,} bytes ({human_bytes(output_size)})**",
        f"- 원본 SHA-256: `{source_hash}`",
        f"- 새 CSV SHA-256: `{output_hash}`",
        "- 원본 보존 검증: **통과(SHA-256 작업 전후 동일)**",
        "",
        "## 중요: CSV와 데이터 타입",
        "",
        "CSV 파일은 셀 값만 저장하며 pandas 타입 메타데이터는 저장하지 않습니다. 따라서 새 CSV에는 "
        "권장 타입에 맞춘 표준 텍스트 표현을 기록했고, 이 스크립트의 `load_typed_csv()` 함수가 "
        "`Int64`, `string`, `category`, `period[M]`, `datetime64[ns]` 타입을 명시적으로 복원합니다.",
        "",
        "## 컬럼별 적용 타입 및 변환 결과",
        "",
        "| 컬럼 | 기존 보고서 권장 타입 | 적용 pandas 타입 | 새 CSV 표현 | 변환 규칙 | 변환 후 결측 |",
        "|---|---|---|---|---|---:|",
    ]
    for column, recommended, applied, representation, rule in TYPE_DETAILS:
        lines.append(
            f"| {column} | {recommended} | `{applied}` | {representation} | {rule} | "
            f"{null_counts[column]:,} |"
        )

    lines.extend(["", "## 적용 타입 개수", ""])
    for dtype in ("Int64", "string", "float64", "period[M]", "category", "datetime64[ns]"):
        lines.append(f"- `{dtype}`: {dtype_counts[dtype]}개 컬럼")

    lines.extend(
        [
            "",
            "## 검증 결과",
            "",
            f"- 행 수 유지: **통과({row_count:,}행)**",
            f"- 컬럼 수 및 순서 유지: **통과({len(EXPECTED_COLUMNS)}열)**",
            "- 숫자 변환 실패값: **0개**",
            "- 날짜 변환 실패값: **0개**",
            "- `본번`, `부번` 4자리 및 선행 0 보존: **통과**",
            "- `거래금액(만원)` 쉼표 제거 및 정수화: **통과**",
            "- `계약년월`의 `YYYY-MM` 표준화: **통과**",
            "- 날짜 컬럼의 `YYYY-MM-DD` 표준화와 `-` 결측 처리: **통과**",
            "- 원본 파일 미변경: **통과**",
            "",
            "## 권장 타입으로 불러오기",
            "",
            "같은 폴더에서 다음과 같이 불러오면 권장 pandas 타입이 복원됩니다.",
            "",
            "```python",
            "from 아파트_실거래가_시계열통합_권장타입적용 import load_typed_csv",
            "",
            "df = load_typed_csv()",
            "print(df.dtypes)",
            "```",
            "",
            "## 전체 작업 재실행",
            "",
            "```powershell",
            f"python .\\{Path(__file__).name}",
            "```",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {SOURCE_CSV}")
    if SOURCE_CSV.resolve() == OUTPUT_CSV.resolve():
        raise RuntimeError("출력 경로가 원본 경로와 같아 작업을 중단합니다.")

    source_hash_before = sha256_file(SOURCE_CSV)
    row_count = 0
    null_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()

    try:
        with TEMP_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:
            first_chunk = True
            reader = pd.read_csv(
                SOURCE_CSV,
                encoding="utf-8-sig",
                dtype="string",
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            )
            for raw_chunk in reader:
                converted = transform_source_chunk(raw_chunk)
                if first_chunk:
                    dtype_counts.update(str(dtype) for dtype in converted.dtypes)
                null_counts.update({column: int(converted[column].isna().sum()) for column in EXPECTED_COLUMNS})
                # Period[M]은 date_format을 함께 쓰면 월말 날짜로 직렬화될 수 있으므로
                # CSV 기록본에서는 명시적으로 YYYY-MM 문자열로 바꾼다.
                serialized = converted.copy()
                serialized["계약년월"] = converted["계약년월"].astype("string")
                serialized.to_csv(
                    output_file,
                    index=False,
                    header=first_chunk,
                    date_format="%Y-%m-%d",
                )
                row_count += len(converted)
                first_chunk = False

        validated_rows = 0
        for serialized_chunk in pd.read_csv(
            TEMP_CSV,
            encoding="utf-8-sig",
            dtype="string",
            chunksize=CHUNK_SIZE,
        ):
            validate_serialized_chunk(serialized_chunk)
            validated_rows += len(serialized_chunk)
        if validated_rows != row_count:
            raise ValueError(f"행 수 검증 실패: 기록 {row_count:,}, 재확인 {validated_rows:,}")

        source_hash_after = sha256_file(SOURCE_CSV)
        if source_hash_before != source_hash_after:
            raise RuntimeError("원본 파일 해시가 작업 중 변경되었습니다.")

        TEMP_CSV.replace(OUTPUT_CSV)
        output_hash = sha256_file(OUTPUT_CSV)
        write_report(
            source_hash=source_hash_before,
            row_count=row_count,
            null_counts=null_counts,
            dtype_counts=dtype_counts,
            output_hash=output_hash,
        )
    finally:
        TEMP_CSV.unlink(missing_ok=True)

    print(f"변환 완료: {row_count:,}행 × {len(EXPECTED_COLUMNS)}열")
    print(f"새 CSV: {OUTPUT_CSV}")
    print(f"보고서: {REPORT_MD}")


if __name__ == "__main__":
    main()
