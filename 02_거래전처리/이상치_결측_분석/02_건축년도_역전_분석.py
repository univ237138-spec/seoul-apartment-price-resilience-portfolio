from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE_NAME = "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_취소거래제거.csv"
OUTPUT_NAME = "건축년도_역전행.csv"
SUMMARY_NAME = "건축년도_역전_요약.json"


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


def analyze(source: Path, output: Path, summary_path: Path) -> dict:
    df = pd.read_csv(source, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {"NO", "시군구", "번지", "단지명", "계약년월", "건축년도"}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")

    contract_year = pd.to_numeric(
        df["계약년월"].str.extract(r"^(\d{4})", expand=False), errors="coerce"
    ).astype("Int64")
    building_year = pd.to_numeric(df["건축년도"], errors="coerce").astype("Int64")
    valid = contract_year.notna() & building_year.notna()
    year_gap = building_year - contract_year
    reversed_mask = valid & year_gap.gt(0)
    selected_index = df.index[reversed_mask]

    diagnostic = pd.DataFrame(
        {
            "_원본CSV행번호": selected_index + 2,
            "_판정": "건축년도 역전 후보",
            "_계약연도": contract_year.loc[selected_index].astype(int),
            "_건축년도차이_년": year_gap.loc[selected_index].astype(int),
        },
        index=selected_index,
    )
    result = pd.concat([diagnostic, df.loc[selected_index].copy()], axis=1)
    result = result.sort_values(
        ["_계약연도", "시군구", "번지", "단지명", "계약년월", "계약일", "_원본CSV행번호"],
        kind="stable",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    year_counts = (
        contract_year.loc[selected_index]
        .value_counts()
        .sort_index()
        .rename_axis("year")
    )
    gap_counts = (
        year_gap.loc[selected_index].value_counts().sort_index().rename_axis("gap")
    )
    complex_counts = (
        result.groupby(["시군구", "번지", "단지명"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    complexes = [
        {"시군구": key[0], "번지": key[1], "단지명": key[2], "행수": int(value)}
        for key, value in complex_counts.items()
    ]

    summary = {
        "analysis": "건축년도 역전",
        "source_file": source.name,
        "source_sha256": file_sha256(source),
        "source_rows": int(len(df)),
        "source_columns": int(len(df.columns)),
        "valid_year_rows": int(valid.sum()),
        "invalid_contract_year_rows": int(contract_year.isna().sum()),
        "invalid_building_year_rows": int(building_year.isna().sum()),
        "flagged_rows": int(len(result)),
        "distinct_complexes": int(len(complex_counts)),
        "minimum_gap_years": int(year_gap.loc[selected_index].min()) if len(result) else None,
        "maximum_gap_years": int(year_gap.loc[selected_index].max()) if len(result) else None,
        "by_contract_year": {str(int(k)): int(v) for k, v in year_counts.items()},
        "by_gap_years": {str(int(k)): int(v) for k, v in gap_counts.items()},
        "by_complex": complexes,
        "criteria": "건축년도 > 계약년월에서 추출한 계약연도",
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
    parser = argparse.ArgumentParser(description="건축년도 역전 후보 분석")
    parser.add_argument("--input", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary", type=Path, default=default_summary)
    args = parser.parse_args()
    summary = analyze(args.input.resolve(), args.output.resolve(), args.summary.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
