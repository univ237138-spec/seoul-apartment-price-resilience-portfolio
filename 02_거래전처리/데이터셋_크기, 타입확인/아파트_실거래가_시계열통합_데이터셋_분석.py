"""아파트 실거래가 시계열 통합 CSV의 크기와 컬럼 타입을 분석한다."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parents[1] / "01_원천데이터" / "아파트_기본_DATA_raw" / "아파트(매매)_실거래가_2017_2026_시계열통합.csv"
REPORT_PATH = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합_분석결과.md"


SEMANTIC_TYPES = {
    "NO": "정수형 식별자(Int64) 또는 문자열",
    "시군구": "문자열(string)",
    "번지": "문자열(string)",
    "본번": "문자열(string, 선행 0 보존)",
    "부번": "문자열(string, 선행 0 보존)",
    "단지명": "문자열(string)",
    "전용면적(㎡)": "실수형(float64)",
    "계약년월": "연월형(YYYYMM; 문자열 또는 Period[M])",
    "계약일": "정수형(Int64)",
    "거래금액(만원)": "정수형(Int64)",
    "동": "문자열(string)",
    "층": "정수형(Int64)",
    "매수자": "범주/문자열(category/string)",
    "매도자": "범주/문자열(category/string)",
    "건축년도": "정수형(Int64)",
    "도로명": "문자열(string)",
    "해제사유발생일": "날짜형(datetime64[ns])",
    "거래유형": "범주/문자열(category/string)",
    "중개사소재지": "문자열(string)",
    "등기일자": "날짜형(datetime64[ns])",
}


def human_bytes(size: int) -> str:
    """바이트 수를 읽기 쉬운 이진 단위로 변환한다."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.2f} {unit}"
        value /= 1024
    raise AssertionError("도달할 수 없는 코드")


def markdown_escape(value: object) -> str:
    """Markdown 표 안에 안전하게 값을 표시한다."""
    if pd.isna(value):
        return "(결측)"
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"분석할 파일을 찾을 수 없습니다: {CSV_PATH}")

    # UTF-8 BOM을 제거해 첫 컬럼명이 '\ufeffNO'가 되지 않도록 utf-8-sig를 사용한다.
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)

    row_count, column_count = df.shape
    file_size = CSV_PATH.stat().st_size
    memory_bytes = int(df.memory_usage(index=True, deep=True).sum())
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()

    schema_rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        null_count = int(series.isna().sum())
        non_null = series.dropna()
        example = non_null.iloc[0] if not non_null.empty else pd.NA
        schema_rows.append(
            {
                "컬럼": column,
                "pandas 추론 타입": str(series.dtype),
                "분석 권장 타입": SEMANTIC_TYPES.get(column, "원본 의미 확인 필요"),
                "비결측": int(series.notna().sum()),
                "결측": null_count,
                "결측률": null_count / row_count if row_count else 0.0,
                "고유값": int(series.nunique(dropna=True)),
                "예시": example,
            }
        )

    lines = [
        "# 아파트(매매) 실거래가 시계열 통합 데이터셋 분석",
        "",
        f"- 분석 시각: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 원본 파일: `{CSV_PATH.name}`",
        f"- 인코딩: UTF-8 with BOM (`utf-8-sig`)",
        f"- 파일 크기: **{file_size:,} bytes ({human_bytes(file_size)})**",
        f"- 데이터 크기(shape): **{row_count:,}행 × {column_count:,}열**",
        f"- 총 데이터 셀 수: **{row_count * column_count:,}개**",
        f"- pandas 메모리 사용량(deep): **{memory_bytes:,} bytes ({human_bytes(memory_bytes)})**",
        f"- 중복 컬럼명: **{'없음' if not duplicate_columns else ', '.join(duplicate_columns)}**",
        "",
        "## 컬럼별 타입 및 품질 요약",
        "",
        "`pandas 추론 타입`은 CSV를 별도 dtype 지정 없이 전체 로드했을 때의 실제 결과입니다. "
        "`분석 권장 타입`은 날짜 연산과 식별자 선행 0 보존 등을 고려한 권장값입니다.",
        "",
        "| 컬럼 | pandas 추론 타입 | 분석 권장 타입 | 비결측 | 결측 | 결측률 | 고유값 | 첫 비결측 예시 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]

    for item in schema_rows:
        lines.append(
            "| {컬럼} | {pandas 추론 타입} | {분석 권장 타입} | {비결측:,} | {결측:,} | "
            "{결측률:.2%} | {고유값:,} | {예시} |".format(
                **{**item, "예시": markdown_escape(item["예시"])}
            )
        )

    lines.extend(
        [
            "",
            "## pandas 타입 개수",
            "",
        ]
    )
    for dtype, count in df.dtypes.astype(str).value_counts().items():
        lines.append(f"- `{dtype}`: {count}개 컬럼")

    lines.extend(
        [
            "",
            "## 해석 시 주의사항",
            "",
            "- `object`는 CSV에서 일반적으로 문자열 또는 혼합값을 뜻합니다. 후속 분석 시 pandas의 "
            "`string` 타입으로 명시하면 결측값 처리가 더 일관적입니다.",
            "- `본번`, `부번`은 숫자처럼 보여도 선행 0이 의미 있으므로 문자열 유지가 안전합니다.",
            "- `계약년월`, `해제사유발생일`, `등기일자`는 원본 저장 타입과 별개로 날짜/연월 타입으로 "
            "변환한 뒤 시계열 분석하는 것이 좋습니다.",
            "- 결측값 때문에 정수 컬럼이 `float64`로 추론될 수 있습니다. 정수 의미를 유지하려면 nullable "
            "integer인 `Int64`를 권장합니다.",
            "",
            "## 재현 방법",
            "",
            "같은 폴더에서 아래 명령을 실행하면 이 보고서를 다시 생성합니다.",
            "",
            "```powershell",
            f"python .\\{Path(__file__).name}",
            "```",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"분석 완료: {row_count:,}행 × {column_count:,}열")
    print(f"보고서 저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
