from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE_NAME = "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_취소거래제거.csv"
MAPPING_NAME = "비구조적_결측행_도로명주소_매핑완료.csv"
OUTPUT_NAME = "아파트(매매)_실거래가_2017_2026_권장타입적용_취소거래제거_도로명결측보완.csv"
SUMMARY_NAME = "도로명주소_매핑적용_요약.json"

ROW_KEY = "_원본CSV행번호"
TARGET_COLUMN = "도로명"
EXPECTED_MAPPING_ROWS = 205
MANUAL_REVIEW_ROWS = {
    602075: "포레나노원: 동 미기재, 단지에 도로명 2개 존재",
    600918: "중앙하이츠: 동 미기재, 단지에 도로명 여러 개 존재",
    601985: "중앙하이츠: 동 미기재, 단지에 도로명 여러 개 존재",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_missing(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin(
        {"", "-", "nan", "none", "null", "na", "n/a"}
    )


def default_paths() -> tuple[Path, Path, Path, Path]:
    analysis_dir = Path(__file__).resolve().parent
    source = analysis_dir.parent / SOURCE_NAME
    mapping = analysis_dir / MAPPING_NAME
    output = analysis_dir / OUTPUT_NAME
    summary = analysis_dir / SUMMARY_NAME
    return source, mapping, output, summary


def apply_mapping(
    source_path: Path,
    mapping_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict:
    source_hash_before = file_sha256(source_path)
    source = pd.read_csv(
        source_path, encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    mapping = pd.read_csv(
        mapping_path, encoding="utf-8-sig", dtype=str, keep_default_na=False
    )

    required_mapping_columns = {ROW_KEY, TARGET_COLUMN, *source.columns}
    missing_mapping_columns = sorted(required_mapping_columns.difference(mapping.columns))
    if missing_mapping_columns:
        raise ValueError(f"매핑 파일 필수 컬럼 누락: {missing_mapping_columns}")
    if len(mapping) != EXPECTED_MAPPING_ROWS:
        raise ValueError(
            f"매핑 행 수가 {EXPECTED_MAPPING_ROWS}행이 아닙니다: {len(mapping)}행"
        )

    row_numbers = pd.to_numeric(mapping[ROW_KEY], errors="raise").astype(int)
    if row_numbers.duplicated().any():
        duplicates = row_numbers[row_numbers.duplicated(keep=False)].tolist()
        raise ValueError(f"매핑 파일의 원본 행번호 중복: {duplicates}")
    if row_numbers.lt(2).any() or row_numbers.gt(len(source) + 1).any():
        raise ValueError("매핑 파일에 원본 CSV 범위를 벗어난 행번호가 있습니다.")

    source_positions = row_numbers.to_numpy() - 2
    source_rows = source.iloc[source_positions].copy()
    source_rows.index = row_numbers.to_numpy()
    mapping_rows = mapping.copy()
    mapping_rows.index = row_numbers.to_numpy()

    comparison_columns = [column for column in source.columns if column != TARGET_COLUMN]
    mismatch_counts = {
        column: int(source_rows[column].ne(mapping_rows[column]).sum())
        for column in comparison_columns
        if source_rows[column].ne(mapping_rows[column]).any()
    }
    if mismatch_counts:
        raise ValueError(f"원본 행과 매핑 행의 도로명 외 값 불일치: {mismatch_counts}")

    source_missing_mask = is_missing(source_rows[TARGET_COLUMN])
    if not source_missing_mask.all():
        bad_rows = source_rows.index[~source_missing_mask].tolist()
        raise ValueError(f"원본에서 도로명이 이미 존재하는 매핑 대상: {bad_rows}")
    if is_missing(mapping_rows[TARGET_COLUMN]).any():
        bad_rows = mapping_rows.index[is_missing(mapping_rows[TARGET_COLUMN])].tolist()
        raise ValueError(f"매핑 파일의 도로명이 비어 있는 행: {bad_rows}")

    absent_review_rows = sorted(set(MANUAL_REVIEW_ROWS).difference(row_numbers))
    if absent_review_rows:
        raise ValueError(f"수동 확인 대상 행이 매핑 파일에 없습니다: {absent_review_rows}")

    completed = source.copy()
    target_column_index = completed.columns.get_loc(TARGET_COLUMN)
    completed.iloc[source_positions, target_column_index] = mapping_rows[
        TARGET_COLUMN
    ].to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed.to_csv(output_path, index=False, encoding="utf-8-sig")

    verified = pd.read_csv(
        output_path, encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    if verified.shape != source.shape:
        raise ValueError(f"출력 크기 불일치: 원본 {source.shape}, 출력 {verified.shape}")
    if list(verified.columns) != list(source.columns):
        raise ValueError("출력 CSV의 컬럼 순서가 원본과 다릅니다.")

    non_target_equal = verified[comparison_columns].equals(source[comparison_columns])
    if not non_target_equal:
        raise ValueError("출력 CSV에서 도로명 외 컬럼이 변경되었습니다.")
    changed_mask = source[TARGET_COLUMN].ne(verified[TARGET_COLUMN])
    changed_positions = set(source.index[changed_mask])
    expected_positions = set(source_positions.tolist())
    if changed_positions != expected_positions:
        raise ValueError("도로명이 변경된 행과 매핑 대상 행이 일치하지 않습니다.")
    verified_mapping_values = verified.iloc[source_positions][TARGET_COLUMN].to_numpy()
    if not (verified_mapping_values == mapping_rows[TARGET_COLUMN].to_numpy()).all():
        raise ValueError("출력 CSV의 도로명이 매핑 파일의 값과 일치하지 않습니다.")

    source_hash_after = file_sha256(source_path)
    if source_hash_before != source_hash_after:
        raise ValueError("작업 중 원본 CSV의 SHA-256이 변경되었습니다.")

    status_counts = (
        mapping["도로명_웹매핑상태"].value_counts(dropna=False).to_dict()
        if "도로명_웹매핑상태" in mapping.columns
        else {}
    )
    road_counts = (
        mapping[TARGET_COLUMN].str.strip().value_counts().sort_index().to_dict()
    )
    summary = {
        "source_file": source_path.name,
        "source_rows": int(len(source)),
        "source_columns": int(len(source.columns)),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "mapping_file": mapping_path.name,
        "mapping_rows": int(len(mapping)),
        "mapping_sha256": file_sha256(mapping_path),
        "output_file": output_path.name,
        "output_rows": int(len(verified)),
        "output_columns": int(len(verified.columns)),
        "output_size_bytes": int(output_path.stat().st_size),
        "output_sha256": file_sha256(output_path),
        "changed_rows": int(changed_mask.sum()),
        "changed_column": TARGET_COLUMN,
        "non_target_columns_unchanged": bool(non_target_equal),
        "road_missing_before": int(is_missing(source[TARGET_COLUMN]).sum()),
        "road_missing_after": int(is_missing(verified[TARGET_COLUMN]).sum()),
        "mapping_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "mapped_road_counts": {str(k): int(v) for k, v in road_counts.items()},
        "manual_review_rows": [
            {
                "original_csv_row": row_number,
                "mapped_road": mapping_rows.loc[row_number, TARGET_COLUMN],
                "reason": reason,
            }
            for row_number, reason in MANUAL_REVIEW_ROWS.items()
        ],
        "validation": {
            "mapping_row_count_205": True,
            "mapping_row_key_unique": True,
            "source_mapping_non_road_values_match": True,
            "only_road_column_changed": True,
            "all_mapping_values_applied": True,
            "source_unchanged": True,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    default_source, default_mapping, default_output, default_summary = default_paths()
    parser = argparse.ArgumentParser(description="도로명 결측 205행 매핑 적용")
    parser.add_argument("--input", type=Path, default=default_source)
    parser.add_argument("--mapping", type=Path, default=default_mapping)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary", type=Path, default=default_summary)
    args = parser.parse_args()
    summary = apply_mapping(
        args.input.resolve(),
        args.mapping.resolve(),
        args.output.resolve(),
        args.summary.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
