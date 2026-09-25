from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_NAME = "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_취소거래제거.csv"
OUTPUT_NAME = "조건부_가격_이상치행.csv"
SUMMARY_NAME = "조건부_가격_이상치_요약.json"

MIN_GROUP_SIZE = 5
ROBUST_Z_LIMIT = 5.0
LOW_RATIO_LIMIT = 0.5
HIGH_RATIO_LIMIT = 2.0


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
    required = {
        "NO",
        "시군구",
        "번지",
        "단지명",
        "전용면적(㎡)",
        "계약년월",
        "거래금액(만원)",
    }
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")

    area = pd.to_numeric(df["전용면적(㎡)"], errors="coerce")
    price = pd.to_numeric(
        df["거래금액(만원)"].str.replace(",", "", regex=False), errors="coerce"
    )
    contract_year = pd.to_numeric(
        df["계약년월"].str.extract(r"^(\d{4})", expand=False), errors="coerce"
    ).astype("Int64")
    area_band = np.floor(area + 0.5).astype("Int64")
    price_per_sqm = price / area

    valid = (
        contract_year.notna()
        & area.notna()
        & price.notna()
        & area.gt(0)
        & price.gt(0)
    )
    work = df.loc[valid, ["시군구", "번지", "단지명"]].copy()
    work["_계약연도"] = contract_year.loc[valid]
    work["_면적구간"] = area_band.loc[valid]
    work["_㎡당가격"] = price_per_sqm.loc[valid]

    group_keys = ["시군구", "번지", "단지명", "_계약연도", "_면적구간"]
    grouped = work.groupby(group_keys, dropna=False, sort=False)["_㎡당가격"]
    group_size = grouped.transform("count")
    group_median = grouped.transform("median")
    absolute_deviation = (work["_㎡당가격"] - group_median).abs()
    group_mad = absolute_deviation.groupby(
        [work[column] for column in group_keys], dropna=False, sort=False
    ).transform("median")

    robust_z = pd.Series(0.0, index=work.index)
    positive_mad = group_mad.gt(0)
    robust_z.loc[positive_mad] = (
        0.6745 * absolute_deviation.loc[positive_mad] / group_mad.loc[positive_mad]
    )
    zero_mad_difference = group_mad.eq(0) & absolute_deviation.gt(0)
    robust_z.loc[zero_mad_difference] = np.inf
    median_ratio = work["_㎡당가격"] / group_median

    eligible = group_size.ge(MIN_GROUP_SIZE)
    flagged = (
        eligible
        & robust_z.ge(ROBUST_Z_LIMIT)
        & (
            median_ratio.le(LOW_RATIO_LIMIT)
            | median_ratio.ge(HIGH_RATIO_LIMIT)
        )
    )

    flagged_index = work.index[flagged]
    result = df.loc[flagged_index].copy()
    direction = np.where(
        median_ratio.loc[flagged_index].le(LOW_RATIO_LIMIT), "낮은 가격", "높은 가격"
    )
    robust_z_text = robust_z.loc[flagged_index].map(
        lambda value: "INF" if np.isinf(value) else f"{value:.2f}"
    )
    diagnostic = pd.DataFrame(
        {
            "_원본CSV행번호": flagged_index + 2,
            "_판정": "조건부 가격 이상치 후보",
            "_이상치방향": direction,
            "_비교계약연도": work.loc[flagged_index, "_계약연도"].astype(int),
            "_비교면적구간_반올림1㎡": work.loc[
                flagged_index, "_면적구간"
            ].astype(int),
            "_비교그룹행수": group_size.loc[flagged_index].astype(int),
            "_㎡당가격_만원": work.loc[flagged_index, "_㎡당가격"].round(2),
            "_그룹중앙값_㎡당만원": group_median.loc[flagged_index].round(2),
            "_중앙값대비배율": median_ratio.loc[flagged_index].round(4),
            "_강건Z점수": robust_z_text,
        },
        index=flagged_index,
    )
    result = pd.concat([diagnostic, result], axis=1)
    result = result.sort_values(
        ["_이상치방향", "_중앙값대비배율", "_원본CSV행번호"],
        ascending=[True, True, True],
        kind="stable",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    group_frame = work.loc[eligible, group_keys].drop_duplicates()
    flagged_groups = work.loc[flagged, group_keys].drop_duplicates()
    year_counts = (
        work.loc[flagged, "_계약연도"]
        .value_counts()
        .sort_index()
        .rename_axis("year")
    )

    def sample_records(frame: pd.DataFrame) -> list[dict]:
        columns = [
            "_원본CSV행번호",
            "시군구",
            "번지",
            "단지명",
            "전용면적(㎡)",
            "계약년월",
            "계약일",
            "거래금액(만원)",
            "층",
            "_중앙값대비배율",
            "_강건Z점수",
        ]
        return frame.loc[:, columns].to_dict(orient="records")

    low = result[result["_이상치방향"].eq("낮은 가격")]
    high = result[result["_이상치방향"].eq("높은 가격")]
    summary = {
        "analysis": "조건부 가격 이상치",
        "source_file": source.name,
        "source_sha256": file_sha256(source),
        "source_rows": int(len(df)),
        "source_columns": int(len(df.columns)),
        "valid_numeric_rows": int(valid.sum()),
        "eligible_rows": int(eligible.sum()),
        "eligible_groups": int(len(group_frame)),
        "flagged_rows": int(len(result)),
        "flagged_groups": int(len(flagged_groups)),
        "low_price_rows": int(len(low)),
        "high_price_rows": int(len(high)),
        "infinite_robust_z_rows": int(np.isinf(robust_z.loc[flagged_index]).sum()),
        "by_contract_year": {str(int(k)): int(v) for k, v in year_counts.items()},
        "criteria": {
            "comparison_group": "시군구+번지+단지명+계약연도+전용면적 반올림 1㎡ 구간",
            "minimum_group_size": MIN_GROUP_SIZE,
            "robust_z_minimum": ROBUST_Z_LIMIT,
            "median_ratio_low_maximum": LOW_RATIO_LIMIT,
            "median_ratio_high_minimum": HIGH_RATIO_LIMIT,
        },
        "lowest_ratio_samples": sample_records(low.nsmallest(5, "_중앙값대비배율")),
        "highest_ratio_samples": sample_records(high.nlargest(5, "_중앙값대비배율")),
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
    parser = argparse.ArgumentParser(description="조건부 가격 이상치 후보 분석")
    parser.add_argument("--input", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--summary", type=Path, default=default_summary)
    args = parser.parse_args()
    summary = analyze(args.input.resolve(), args.output.resolve(), args.summary.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
