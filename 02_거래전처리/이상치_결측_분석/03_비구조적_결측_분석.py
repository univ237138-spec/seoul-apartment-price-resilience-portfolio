from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE_NAME = "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_취소거래제거.csv"
OUTPUT_NAME = "비구조적_결측행.csv"
SUMMARY_NAME = "비구조적_결측_요약.json"

CORE_COLUMNS = [
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
    "층",
    "건축년도",
    "도로명",
]
STRUCTURALLY_OPTIONAL_COLUMNS = [
    "동",
    "매수자",
    "매도자",
    "해제사유발생일",
    "거래유형",
    "중개사소재지",
    "등기일자",
]
MISSING_TOKENS = {"", "-", "nan", "none", "null", "na", "n/a"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_paths() -> tuple[Path, Path, Path]:
    output_dir = Path(__file__).resolve().parent
    source = output_dir.parent / SOURCE_NAME
    return source, output_dir / OUTPUT_NAME, output_dir / SUMMARY_NAME


def normalized_missing(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin(MISSING_TOKENS)


def analyze(source: Path, output: Path, summary_path: Path) -> dict:
    df = pd.read_csv(source, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing_columns = [column for column in CORE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"핵심 분석 컬럼 누락: {missing_columns}")

    missing_matrix = pd.DataFrame(
        {column: normalized_missing(df[column]) for column in CORE_COLUMNS},
        index=df.index,
    )
    missing_count = missing_matrix.sum(axis=1)
    flagged = missing_count.gt(0)
    selected_index = df.index[flagged]

    missing_columns_text = missing_matrix.loc[selected_index].apply(
        lambda row: ", ".join(row.index[row].tolist()), axis=1
    )
    diagnostic = pd.DataFrame(
        {
            "_원본CSV행번호": selected_index + 2,
            "_판정": "비구조적 결측 후보",
            "_결측컬럼수": missing_count.loc[selected_index].astype(int),
            "_결측컬럼목록": missing_columns_text,
        },
        index=selected_index,
    )
    result = pd.concat([diagnostic, df.loc[selected_index].copy()], axis=1)
    result = result.sort_values(
        ["_결측컬럼목록", "계약년월", "시군구", "번지", "단지명", "_원본CSV행번호"],
        kind="stable",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    contract_year = pd.to_numeric(
        df["계약년월"].str.extract(r"^(\d{4})", expand=False), errors="coerce"
    ).astype("Int64")
    year_counts = (
        contract_year.loc[selected_index]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("year")
    )
    complex_counts = (
        result.groupby(["시군구", "번지", "단지명"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    top_complexes = [
        {"시군구": key[0], "번지": key[1], "단지명": key[2], "행수": int(value)}
        for key, value in complex_counts.head(15).items()
    ]

    summary = {
        "analysis": "비구조적 결측",
        "source_file": source.name,
        "source_sha256": file_sha256(source),
        "source_rows": int(len(df)),
        "source_columns": int(len(df.columns)),
        "flagged_rows": int(len(result)),
        "missing_cells": int(missing_matrix.to_numpy().sum()),
        "rows_with_multiple_missing_core_columns": int(missing_count.gt(1).sum()),
        "missing_by_core_column": {
            column: int(value) for column, value in missing_matrix.sum().items()
        },
        "by_contract_year": {
            ("미상" if pd.isna(k) else str(int(k))): int(v) for k, v in year_counts.items()
        },
        "distinct_complexes": int(len(complex_counts)),
        "top_complexes": top_complexes,
        "core_columns_checked": CORE_COLUMNS,
        "structurally_optional_columns_excluded": STRUCTURALLY_OPTIONAL_COLUMNS,
        "missing_tokens": sorted(MISSING_TOKENS),
        "criteria": "핵심 컬럼에서 공백·공백문자·하이픈·대표 결측 문자열 탐지",
        "output_file": output.name,
        "output_rows": int(len(pd.read_csv(output, encoding="utf-8-sig"))),
        "output_sha256": file_sha256(output),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    default_source, default_output, default_summary = default_paths()
    parser = argparse.ArgumentParser(description="비구조적 결측 후보 분석")
    parser.add_argument("--input", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary", type=Path, default=default_summary)
    args = parser.parse_args()
    summary = analyze(args.input.resolve(), args.output.resolve(), args.summary.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
