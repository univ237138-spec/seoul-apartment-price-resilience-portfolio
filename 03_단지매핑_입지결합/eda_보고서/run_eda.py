from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "01_원천데이터" / "아파트_기본_DATA_raw" / "아파트(매매)_실거래가_2017_2026_시계열통합.csv"
COMPLEX = ROOT / "단지정보_데이터_clean.csv"
LOCATION = ROOT / "01_원천데이터" / "아파트_입지_DATA.csv"
OUTPUT = ROOT / "04_EDA" / "eda"
FIGURES = OUTPUT / "figures"
SOURCE_NAME = INPUT.name
MISSING_TOKENS = {"", "-", "na", "n/a", "null", "none", "nan"}


def log(message: str) -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig")


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-_/().,]+", "", regex=True)
    )


def numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.mask(cleaned.str.lower().isin(MISSING_TOKENS))
    return pd.to_numeric(cleaned, errors="coerce")


def missing_mask(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin(MISSING_TOKENS)


def file_inventory() -> pd.DataFrame:
    rows: list[dict] = []
    candidates: list[Path] = []
    for p in ROOT.rglob("*.csv"):
        rel = p.relative_to(ROOT)
        rel_text = str(rel)
        if rel.parts and rel.parts[0] in {"04_EDA", "05_발표자료"}:
            continue
        if "중복분석_2017_2026" in rel_text:
            continue
        candidates.append(p)
    for p in sorted(candidates, key=lambda x: str(x)):
        encoding = "utf-8-sig"
        header_row = 0
        sample = None
        read_error = ""
        for candidate_encoding in ["utf-8-sig", "cp949"]:
            try:
                with p.open("r", encoding=candidate_encoding) as preview_file:
                    preview_lines = [preview_file.readline() for _ in range(40)]
                comma_counts = [line.count(",") for line in preview_lines]
                header_row = comma_counts.index(max(comma_counts)) if comma_counts else 0
                sample = pd.read_csv(
                    p,
                    nrows=200,
                    encoding=candidate_encoding,
                    dtype=str,
                    keep_default_na=False,
                    skiprows=header_row,
                )
                encoding = candidate_encoding
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                read_error = str(exc)
        if sample is None:
            rows.append(
                {
                    "파일명": str(p.relative_to(ROOT)),
                    "행수": np.nan,
                    "열수": np.nan,
                    "인코딩": "UNRESOLVED",
                    "예상_grain": "확인 필요",
                    "컬럼명": "",
                    "파일크기_bytes": p.stat().st_size,
                    "표본200행_결측포함행": np.nan,
                    "PK_후보": "",
                    "Join_Key_후보": "",
                    "inventory_error": read_error,
                }
            )
            continue
        with p.open("rb") as f:
            line_count = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
        row_count = max(line_count - header_row - 1, 0)
        sample_missing = int(sample.apply(missing_mask).any(axis=1).sum()) if len(sample) else 0
        rows.append(
            {
                "파일명": str(p.relative_to(ROOT)),
                "행수": row_count,
                "열수": len(sample.columns),
                "인코딩": encoding,
                "예상_grain": "아파트 매매 거래 1건" if "실거래가" in p.name or "거래_DATA" in p.name else "참조 엔터티 1건",
                "컬럼명": " | ".join(sample.columns),
                "파일크기_bytes": p.stat().st_size,
                "표본200행_결측포함행": sample_missing,
                "PK_후보": " | ".join(c for c in ["NO", "KAPT단지코드"] if c in sample.columns),
                "Join_Key_후보": " | ".join(c for c in ["시군구", "법정동주소", "도로명주소", "단지명", "위도", "경도"] if c in sample.columns),
                "inventory_error": "",
            }
        )
    return pd.DataFrame(rows)


def setup_plotting() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close()


def robust_stats(series: pd.Series) -> tuple[float, float, float, float, float]:
    valid = series.dropna().astype(float)
    if valid.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    q1, median, q3 = valid.quantile([0.25, 0.5, 0.75])
    mad = float(np.median(np.abs(valid - median)))
    return float(q1), float(median), float(q3), float(q3 - q1), mad


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    before_hash = sha256(INPUT)
    log("[1/10] 데이터 인벤토리")
    inventory = file_inventory()
    save_csv(inventory, "00_dataset_inventory.csv")

    log("[2/10] 원천 데이터 로드 및 추적 열 생성")
    df = pd.read_csv(INPUT, encoding="utf-8-sig", dtype=str, keep_default_na=False, low_memory=False)
    original_cols = df.columns.tolist()
    df["__source_file"] = SOURCE_NAME
    df["__source_row_number"] = np.arange(2, len(df) + 2, dtype=np.int64)
    df["__row_id"] = SOURCE_NAME + ":" + df["__source_row_number"].astype(str)

    num_map = {
        "NO": "__num_NO",
        "본번": "__num_본번",
        "부번": "__num_부번",
        "전용면적(㎡)": "__num_전용면적",
        "계약년월": "__num_계약년월",
        "계약일": "__num_계약일",
        "거래금액(만원)": "__num_거래금액",
        "층": "__num_층",
        "건축년도": "__num_건축년도",
    }
    for source, derived in num_map.items():
        df[derived] = numeric(df[source])
    df["__계약연도"] = (df["__num_계약년월"] // 100).astype("Int64")
    df["__계약월"] = (df["__num_계약년월"] % 100).astype("Int64")
    date_text = (
        df["__계약연도"].astype("string")
        + "-"
        + df["__계약월"].astype("string").str.zfill(2)
        + "-"
        + df["__num_계약일"].astype("Int64").astype("string").str.zfill(2)
    )
    df["__계약일자"] = pd.to_datetime(date_text, errors="coerce")
    df["__분기"] = df["__계약일자"].dt.to_period("Q").astype("string")
    df["__자치구"] = df["시군구"].str.extract(r"서울특별시\s+([^\s]+)", expand=False).fillna("")
    area_bins = [0, 40, 60, 85, 102, 135, np.inf]
    area_labels = ["~40㎡", "40~60㎡", "60~85㎡", "85~102㎡", "102~135㎡", "135㎡~"]
    df["__면적구"] = pd.cut(df["__num_전용면적"], bins=area_bins, labels=area_labels, right=False).astype("string")
    df["__㎡당가격"] = df["__num_거래금액"] / df["__num_전용면적"].replace(0, np.nan)

    log("[3/10] 스키마·결측·중복 진단")
    profile_rows: list[dict] = []
    conversion_rows: list[dict] = []
    missing_masks: dict[str, pd.Series] = {}
    for col in original_cols:
        s = df[col]
        miss = missing_mask(s)
        missing_masks[col] = miss
        nonmissing = s[~miss]
        parsed = numeric(s)
        parse_rate = float(parsed.notna().sum() / max((~miss).sum(), 1))
        inferred = "numeric" if parse_rate >= 0.95 else "string"
        mode = nonmissing.mode().iloc[0] if not nonmissing.empty else ""
        q1 = median = q3 = minimum = maximum = mean = std = np.nan
        if inferred == "numeric" and parsed.notna().any():
            desc = parsed.describe(percentiles=[0.25, 0.5, 0.75])
            minimum, q1, median, q3, maximum = (desc.get(k, np.nan) for k in ["min", "25%", "50%", "75%", "max"])
            mean, std = desc.get("mean", np.nan), desc.get("std", np.nan)
        profile_rows.append(
            {
                "column": col,
                "original_dtype": str(s.dtype),
                "inferred_dtype": inferred,
                "total_count": len(s),
                "non_null_count": int((~miss).sum()),
                "null_count": int(miss.sum()),
                "null_rate": float(miss.mean()),
                "unique_count": int(nonmissing.nunique(dropna=False)),
                "unique_rate": float(nonmissing.nunique(dropna=False) / max(len(nonmissing), 1)),
                "mode": mode,
                "mode_count": int((nonmissing == mode).sum()) if len(nonmissing) else 0,
                "string_min_length": int(nonmissing.str.len().min()) if len(nonmissing) else np.nan,
                "string_max_length": int(nonmissing.str.len().max()) if len(nonmissing) else np.nan,
                "min": minimum,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": maximum,
                "mean": mean,
                "std": std,
                "numeric_parse_rate": parse_rate,
            }
        )
        if parse_rate >= 0.95:
            conversion_rows.append(
                {
                    "column": col,
                    "current_dtype": "object",
                    "numeric_conversion_rate": parse_rate,
                    "recommendation": "천단위 쉼표 제거 후 숫자형 변환 검토",
                    "status": "REVIEW_REQUIRED",
                }
            )
    schema_profile = pd.DataFrame(profile_rows)
    save_csv(schema_profile, "01_schema_profile.csv")
    save_csv(pd.DataFrame(conversion_rows), "01_dtype_conversion_candidates.csv")

    exact_mask = df.duplicated(subset=original_cols, keep=False)
    exact = df.loc[exact_mask, original_cols + ["__source_file", "__source_row_number", "__row_id"]].copy()
    if len(exact):
        exact["duplicate_group_id"] = "EXACT_" + (exact.groupby(original_cols, dropna=False).ngroup() + 1).astype(str)
        exact["status"] = "REVIEW_REQUIRED"
        exact = exact[["duplicate_group_id", "status", "__row_id", "__source_row_number"] + original_cols]
    else:
        exact = pd.DataFrame(columns=["duplicate_group_id", "status", "__row_id", "__source_row_number"] + original_cols)
    save_csv(exact, "02_duplicates_exact_rows.csv")

    duplicate_key_frames: list[pd.DataFrame] = []
    for key in ["NO"]:
        mask = ~missing_masks[key] & df.duplicated(key, keep=False)
        if mask.any():
            temp = (
                df.loc[mask, [key, "__row_id", "__source_row_number"]]
                .groupby(key, dropna=False)
                .agg(
                    count=("__row_id", "size"),
                    related_row_ids=("__row_id", lambda x: " | ".join(x)),
                    source_row_numbers=("__source_row_number", lambda x: " | ".join(x.astype(str))),
                )
                .reset_index()
            )
            temp["key_column"] = key
            temp["key_value"] = temp[key]
            duplicate_key_frames.append(temp[["key_column", "key_value", "count", "related_row_ids", "source_row_numbers"]])
    duplicate_keys = pd.concat(duplicate_key_frames, ignore_index=True) if duplicate_key_frames else pd.DataFrame(
        columns=["key_column", "key_value", "count", "related_row_ids", "source_row_numbers"]
    )
    save_csv(duplicate_keys, "03_duplicate_keys.csv")

    business_key = ["시군구", "단지명", "전용면적(㎡)", "계약년월", "계약일", "거래금액(만원)", "층", "도로명"]
    business_mask = df.duplicated(subset=business_key, keep=False)
    business = df.loc[business_mask, original_cols + ["__source_file", "__source_row_number", "__row_id"]].copy()
    if len(business):
        group_no = business.groupby(business_key, dropna=False).ngroup() + 1
        business["duplicate_candidate_group_id"] = "TX_" + group_no.astype(str)
        business["group_size"] = business.groupby("duplicate_candidate_group_id")["__row_id"].transform("size")
        exact_except_no = business.groupby("duplicate_candidate_group_id")[original_cols].transform("nunique").drop(columns=["NO"]).max(axis=1).eq(1)
        identity_missing = business[["시군구", "단지명", "도로명"]].apply(missing_mask).any(axis=1)
        business["case"] = np.select(
            [identity_missing, exact_except_no],
            ["Case 3: 식별정보 결측", "Case 1: NO 외 핵심정보 동일"],
            default="Case 2: 동일 조건의 별도 거래 가능",
        )
        business["status"] = "REVIEW_REQUIRED"
        business = business[["duplicate_candidate_group_id", "group_size", "case", "status", "__row_id", "__source_row_number"] + original_cols]
    else:
        business = pd.DataFrame(columns=["duplicate_candidate_group_id", "group_size", "case", "status", "__row_id", "__source_row_number"] + original_cols)
    save_csv(business, "04_transaction_duplicate_candidates.csv")

    near_rows: list[dict] = []
    for source_col in ["단지명", "도로명"]:
        norm = normalize_text(df[source_col])
        tmp = pd.DataFrame({"normalized_value": norm, "original_value": df[source_col]}).query("normalized_value != ''")
        variants = tmp.groupby("normalized_value")["original_value"].agg(lambda x: sorted(set(x)))
        variants = variants[variants.map(len) > 1]
        counts = tmp.groupby("normalized_value").size()
        for key, values in variants.items():
            near_rows.append(
                {
                    "column": source_col,
                    "normalized_value": key,
                    "original_variants": " | ".join(values),
                    "variant_count": len(values),
                    "row_count": int(counts[key]),
                    "status": "REVIEW_REQUIRED",
                }
            )
    near = pd.DataFrame(near_rows, columns=["column", "normalized_value", "original_variants", "variant_count", "row_count", "status"])
    save_csv(near, "05_near_duplicate_candidates.csv")

    missing_summary_rows: list[dict] = []
    for col in original_cols:
        stripped = df[col].astype("string").fillna("").str.strip()
        lower = stripped.str.lower()
        token_counts = lower[lower.isin(MISSING_TOKENS)].value_counts(dropna=False)
        token_text = " | ".join(f"{('<blank>' if k == '' else k)}:{int(v)}" for k, v in token_counts.items())
        missing_summary_rows.append(
            {"column": col, "missing_count": int(missing_masks[col].sum()), "missing_rate": float(missing_masks[col].mean()), "missing_token_types": token_text}
        )
    missing_summary = pd.DataFrame(missing_summary_rows)
    save_csv(missing_summary, "06_missing_summary.csv")

    missing_any = pd.Series(False, index=df.index)
    missing_cols_text = pd.Series("", index=df.index, dtype="string")
    structural_cols = {"동", "매수자", "매도자", "해제사유발생일", "거래유형", "중개사소재지", "등기일자"}
    for col in original_cols:
        m = missing_masks[col]
        missing_any |= m
        missing_cols_text = missing_cols_text + np.where(m, np.where(missing_cols_text.eq(""), col, " | " + col), "")
    miss_rows = df.loc[missing_any, original_cols + ["__source_file", "__source_row_number", "__row_id"]].copy()
    miss_rows.insert(0, "missing_columns", missing_cols_text[missing_any].values)
    miss_rows.insert(
        1,
        "missing_semantic_type",
        miss_rows["missing_columns"].map(
            lambda x: "STRUCTURAL_MISSING" if all(c in structural_cols for c in x.split(" | ")) else "POSSIBLE_DATA_QUALITY_ISSUE"
        ),
    )
    miss_rows.insert(2, "status", miss_rows["missing_semantic_type"].map({"STRUCTURAL_MISSING": "STRUCTURAL_MISSING", "POSSIBLE_DATA_QUALITY_ISSUE": "REVIEW_REQUIRED"}))
    save_csv(miss_rows, "07_missing_rows.csv")
    missing_patterns = (
        miss_rows.groupby(["missing_columns", "missing_semantic_type"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )
    save_csv(missing_patterns, "08_missing_patterns.csv")

    log("[4/10] 논리 오류와 통계·조건부 이상치")
    issue_frames: list[pd.DataFrame] = []

    def add_issue(mask: pd.Series, issue_type: str, column: str, reason: str, status: str = "LIKELY_ERROR") -> None:
        if mask.any():
            t = df.loc[mask, ["__row_id", "__source_row_number"]].copy()
            t["issue_type"] = issue_type
            t["column"] = column
            t["current_value"] = df.loc[mask, column].astype(str).values if column in df.columns else ""
            t["reason_flagged"] = reason
            t["status"] = status
            issue_frames.append(t)

    add_issue(df["__num_전용면적"].le(0) | df["__num_전용면적"].isna(), "PHYSICAL_RANGE", "전용면적(㎡)", "숫자 변환 실패 또는 0 이하")
    add_issue(df["__num_거래금액"].le(0) | df["__num_거래금액"].isna(), "PHYSICAL_RANGE", "거래금액(만원)", "숫자 변환 실패 또는 0 이하")
    add_issue(df["__계약일자"].isna(), "INVALID_DATE", "계약년월", "계약년월과 계약일 조합이 유효한 날짜가 아님")
    add_issue(df["__num_건축년도"].gt(df["__계약연도"]), "CROSS_FIELD", "건축년도", "건축년도가 계약연도보다 큼", "REVIEW_REQUIRED")
    add_issue(df["__num_층"].isna(), "FORMAT", "층", "숫자형 변환 실패", "REVIEW_REQUIRED")
    add_issue(df["__num_층"].lt(-10) | df["__num_층"].gt(100), "PHYSICAL_RANGE", "층", "현실성 검토 범위(-10~100층) 밖", "REVIEW_REQUIRED")
    logical = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame(columns=["__row_id", "__source_row_number", "issue_type", "column", "current_value", "reason_flagged", "status"])

    outlier_frames: list[pd.DataFrame] = []
    outlier_summary_rows: list[dict] = []
    metric_map = {
        "전용면적(㎡)": df["__num_전용면적"],
        "층": df["__num_층"],
        "건축년도": df["__num_건축년도"],
        "거래금액(만원)": df["__num_거래금액"],
        "㎡당가격": df["__㎡당가격"],
    }
    for name, values in metric_map.items():
        q1, median, q3, iqr, mad = robust_stats(values)
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        extreme_lower, extreme_upper = q1 - 3 * iqr, q3 + 3 * iqr
        robust_z = 0.6745 * (values - median) / mad if mad and not math.isnan(mad) else pd.Series(np.nan, index=df.index)
        mask = values.lt(lower) | values.gt(upper) | robust_z.abs().gt(3.5)
        extreme = values.lt(extreme_lower) | values.gt(extreme_upper) | robust_z.abs().gt(5)
        if mask.any():
            t = df.loc[mask, ["__row_id", "__source_row_number"]].copy()
            t["outlier_column"] = name
            t["detection_method"] = "GLOBAL_IQR|MAD"
            t["group_condition"] = "전체"
            t["original_value"] = values[mask].values
            t["lower_bound"] = lower
            t["upper_bound"] = upper
            t["robust_z_score"] = robust_z[mask].values
            t["severity"] = np.where(extreme[mask], "HIGH", "MEDIUM")
            outlier_frames.append(t)
        outlier_summary_rows.append(
            {
                "outlier_column": name,
                "detection_method": "GLOBAL_IQR|MAD",
                "group_condition": "전체",
                "q1": q1,
                "median": median,
                "q3": q3,
                "iqr": iqr,
                "mad": mad,
                "lower_bound": lower,
                "upper_bound": upper,
                "candidate_rows": int(mask.sum()),
            }
        )

    group_cols = ["단지명", "__면적구", "__계약연도"]
    group = df.groupby(group_cols, observed=True)["__㎡당가격"]
    g_count = group.transform("count")
    g_median = group.transform("median")
    g_mad = group.transform(lambda x: np.median(np.abs(x - np.median(x))))
    g_robust_z = 0.6745 * (df["__㎡당가격"] - g_median) / g_mad.replace(0, np.nan)
    group_mask = g_count.ge(5) & g_robust_z.abs().gt(3.5)
    if group_mask.any():
        t = df.loc[group_mask, ["__row_id", "__source_row_number"]].copy()
        t["outlier_column"] = "㎡당가격"
        t["detection_method"] = "GROUP_MAD"
        t["group_condition"] = (
            df.loc[group_mask, "단지명"].astype(str) + " | " + df.loc[group_mask, "__면적구"].astype(str) + " | " + df.loc[group_mask, "__계약연도"].astype(str)
        )
        t["original_value"] = df.loc[group_mask, "__㎡당가격"].values
        t["lower_bound"] = (g_median - 3.5 * g_mad / 0.6745)[group_mask].values
        t["upper_bound"] = (g_median + 3.5 * g_mad / 0.6745)[group_mask].values
        t["robust_z_score"] = g_robust_z[group_mask].values
        t["severity"] = np.where(g_robust_z[group_mask].abs().ge(5), "HIGH", "MEDIUM")
        outlier_frames.append(t)
    outlier_summary_rows.append(
        {
            "outlier_column": "㎡당가격",
            "detection_method": "GROUP_MAD",
            "group_condition": "단지명×면적구×계약연도 (n>=5)",
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "mad": np.nan,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "candidate_rows": int(group_mask.sum()),
        }
    )
    outlier_long = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame()
    if len(outlier_long):
        severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        outlier_long["__severity_rank"] = outlier_long["severity"].map(severity_rank)
        agg = outlier_long.groupby("__row_id", sort=False).agg(
            outlier_column=("outlier_column", lambda x: " | ".join(dict.fromkeys(x.astype(str)))),
            detection_method=("detection_method", lambda x: " | ".join(dict.fromkeys(x.astype(str)))),
            group_condition=("group_condition", lambda x: " | ".join(dict.fromkeys(x.astype(str)))),
            original_value=("original_value", lambda x: " | ".join(f"{v:.6g}" if isinstance(v, (int, float, np.number)) else str(v) for v in x)),
            lower_bound=("lower_bound", "min"),
            upper_bound=("upper_bound", "max"),
            robust_z_score=("robust_z_score", lambda x: float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else np.nan),
            severity_rank=("__severity_rank", "max"),
        ).reset_index()
        rank_severity = {v: k for k, v in severity_rank.items()}
        agg["severity"] = agg["severity_rank"].map(rank_severity)
        agg["status"] = "REVIEW_REQUIRED"
        agg["outlier_id"] = [f"OUT_{i:07d}" for i in range(1, len(agg) + 1)]
        outlier_rows = agg.merge(df[original_cols + ["__source_file", "__source_row_number", "__row_id"]], on="__row_id", how="left")
        outlier_rows = outlier_rows.drop(columns="severity_rank")
    else:
        outlier_rows = pd.DataFrame(columns=["outlier_id", "outlier_column", "detection_method", "group_condition", "original_value", "lower_bound", "upper_bound", "robust_z_score", "severity", "status", "__row_id", "__source_row_number"] + original_cols)
    save_csv(pd.DataFrame(outlier_summary_rows), "09_outlier_summary.csv")
    save_csv(outlier_rows, "10_outlier_rows.csv")

    log("[5/10] 시계열 품질·이상 후보·주변 거래")
    q = (
        df.dropna(subset=["__계약일자", "__㎡당가격"])
        .groupby(["단지명", "__면적구", "__분기"], observed=True)
        .agg(transaction_count=("__row_id", "size"), median_unit_price=("__㎡당가격", "median"), median_price=("__num_거래금액", "median"))
        .reset_index()
        .sort_values(["단지명", "__면적구", "__분기"])
    )
    q["prior_rolling_median"] = q.groupby(["단지명", "__면적구"], observed=True)["median_unit_price"].transform(lambda x: x.shift(1).rolling(4, min_periods=3).median())
    q["deviation_from_rolling"] = q["median_unit_price"] / q["prior_rolling_median"] - 1
    q["previous_quarter_change"] = q.groupby(["단지명", "__면적구"], observed=True)["median_unit_price"].pct_change(fill_method=None)
    q["timeseries_outlier"] = (
        q["deviation_from_rolling"].abs().gt(0.5) | q["previous_quarter_change"].abs().gt(0.7)
    ).fillna(False)
    q["status"] = np.where(q["timeseries_outlier"], "REVIEW_REQUIRED", "LIKELY_VALID")
    save_csv(q.loc[q["timeseries_outlier"]].copy(), "12_timeseries_outlier_candidates.csv")

    quality_rows: list[dict] = []
    peak_trough_keys: list[dict] = []
    for (complex_name, area_group), g in q.groupby(["단지명", "__면적구"], observed=True, sort=False):
        periods = pd.PeriodIndex(g["__분기"], freq="Q")
        span_quarters = int(periods.max().ordinal - periods.min().ordinal + 1) if len(periods) else 0
        peak_idx = g["median_unit_price"].idxmax()
        trough_idx = g["median_unit_price"].idxmin()
        quality_rows.append(
            {
                "단지명": complex_name,
                "면적구": area_group,
                "transaction_count": int(g["transaction_count"].sum()),
                "observed_quarters": len(g),
                "first_quarter": str(periods.min()),
                "last_quarter": str(periods.max()),
                "missing_quarters_between": span_quarters - len(g),
                "single_transaction_quarters": int(g["transaction_count"].eq(1).sum()),
                "median_quarter_price": float(g["median_price"].median()),
                "mean_quarter_price": float(g["median_price"].mean()),
                "min_quarter_price": float(g["median_price"].min()),
                "max_quarter_price": float(g["median_price"].max()),
                "quarter_price_iqr": float(g["median_price"].quantile(0.75) - g["median_price"].quantile(0.25)),
                "quarter_price_cv": (
                    float(g["median_price"].std() / g["median_price"].mean())
                    if pd.notna(g["median_price"].std()) and pd.notna(g["median_price"].mean()) and g["median_price"].mean() != 0
                    else np.nan
                ),
                "peak_quarter": g.loc[peak_idx, "__분기"],
                "trough_quarter": g.loc[trough_idx, "__분기"],
                "mdd_reliability": "LOW" if len(g) < 4 or int(g["transaction_count"].sum()) < 8 else "MEDIUM_OR_HIGH",
            }
        )
        if len(g) >= 4 and int(g["transaction_count"].sum()) >= 8:
            peak_trough_keys.extend(
                [
                    {"단지명": complex_name, "__면적구": area_group, "__분기": g.loc[peak_idx, "__분기"], "turning_point": "PEAK"},
                    {"단지명": complex_name, "__면적구": area_group, "__분기": g.loc[trough_idx, "__분기"], "turning_point": "TROUGH"},
                ]
            )
    timeseries_quality = pd.DataFrame(quality_rows)
    save_csv(timeseries_quality, "13_timeseries_quality.csv")
    key_frame = pd.DataFrame(peak_trough_keys)
    if len(key_frame):
        peak_trough = df.merge(key_frame, on=["단지명", "__면적구", "__분기"], how="inner")
        peak_trough = peak_trough[["turning_point", "__분기", "__면적구", "__row_id", "__source_row_number"] + original_cols]
    else:
        peak_trough = pd.DataFrame(columns=["turning_point", "__분기", "__면적구", "__row_id", "__source_row_number"] + original_cols)
    save_csv(peak_trough, "14_peak_trough_source_rows.csv")

    neighbor_targets = outlier_rows.loc[outlier_rows["detection_method"].str.contains("GROUP_MAD", na=False)].copy()
    neighbor_records: list[dict] = []
    if len(neighbor_targets):
        core = df[["__row_id", "단지명", "__면적구", "__계약일자", "__num_거래금액", "__num_전용면적", "__num_층"]].dropna(subset=["__계약일자"]).sort_values("__계약일자")
        grouped = {}
        for (k1, k2), g in core.groupby(["단지명", "__면적구"], observed=True, sort=False):
            grouped[(k1, k2)] = {
                "dates": g["__계약일자"].to_numpy(dtype="datetime64[ns]"),
                "row_ids": g["__row_id"].to_numpy(),
                "prices": g["__num_거래금액"].to_numpy(),
                "areas": g["__num_전용면적"].to_numpy(),
                "floors": g["__num_층"].to_numpy(),
            }
        target_lookup = df.set_index("__row_id")[["단지명", "__면적구", "__계약일자"]]
        for target_id in neighbor_targets["__row_id"]:
            row = target_lookup.loc[target_id]
            group_arrays = grouped.get((row["단지명"], row["__면적구"]))
            if group_arrays is None:
                continue
            target_date = np.datetime64(row["__계약일자"], "ns")
            dates = group_arrays["dates"]
            position = int(np.searchsorted(dates, target_date))
            candidate_idx = np.arange(max(0, position - 8), min(len(dates), position + 9))
            candidate_idx = candidate_idx[group_arrays["row_ids"][candidate_idx] != target_id]
            if not len(candidate_idx):
                continue
            distances = np.abs(dates[candidate_idx].astype("int64") - target_date.astype("int64"))
            nearest_idx = candidate_idx[np.argsort(distances, kind="stable")[:5]]
            for idx in nearest_idx:
                neighbor_records.append(
                    {
                        "target_row_id": target_id,
                        "comparison_group": f"{row['단지명']} | {row['__면적구']}",
                        "comparison_row_id": group_arrays["row_ids"][idx],
                        "comparison_date": dates[idx],
                        "comparison_price": group_arrays["prices"][idx],
                        "comparison_area": group_arrays["areas"][idx],
                        "comparison_floor": group_arrays["floors"][idx],
                    }
                )
    neighbors = pd.DataFrame(neighbor_records, columns=["target_row_id", "comparison_group", "comparison_row_id", "comparison_date", "comparison_price", "comparison_area", "comparison_floor"])
    save_csv(neighbors, "11_outlier_neighbor_rows.csv")

    log("[6/10] 좌표·범주형·교차필드 품질")
    location_df = pd.read_csv(LOCATION, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    location_df["__source_file"] = LOCATION.name
    location_df["__source_row_number"] = np.arange(2, len(location_df) + 2)
    location_df["__row_id"] = LOCATION.name + ":" + location_df["__source_row_number"].astype(str)
    lat = numeric(location_df["위도"])
    lon = numeric(location_df["경도"])
    coord_issue = pd.Series("", index=location_df.index, dtype="string")
    coord_issue = coord_issue.mask(lat.isna() | lon.isna(), "COORDINATE_MISSING")
    coord_issue = coord_issue.mask(lat.notna() & lon.notna() & (~lat.between(37, 38) | ~lon.between(126, 128)), "OUTSIDE_SEOUL_SANITY_RANGE")
    coord_issue = coord_issue.mask(lat.between(126, 128) & lon.between(37, 38), "POSSIBLE_XY_SWAP")
    coord_anomalies = location_df.loc[coord_issue.ne("")].copy()
    coord_anomalies.insert(0, "coordinate_issue_type", coord_issue[coord_issue.ne("")].values)
    coord_anomalies.insert(1, "status", "REVIEW_REQUIRED")
    coord_anomalies["diagnostic_reason"] = "WGS84 위도 37~38, 경도 126~128 범위 점검"
    save_csv(coord_anomalies, "15_coordinate_anomalies.csv")

    categorical_rows: list[dict] = []
    for col in ["시군구", "번지", "단지명", "동", "도로명", "거래유형", "중개사소재지"]:
        s = df[col].astype("string").fillna("")
        categorical_rows.extend(
            [
                {"column": col, "issue_type": "LEADING_OR_TRAILING_WHITESPACE", "row_count": int(s.ne(s.str.strip()).sum()), "status": "REVIEW_REQUIRED"},
                {"column": col, "issue_type": "DASH_TOKEN", "row_count": int(s.str.strip().eq("-").sum()), "status": "STRUCTURAL_MISSING" if col in structural_cols else "REVIEW_REQUIRED"},
                {"column": col, "issue_type": "EMPTY_TOKEN", "row_count": int(s.str.strip().eq("").sum()), "status": "REVIEW_REQUIRED"},
            ]
        )
    categorical = pd.DataFrame(categorical_rows)
    save_csv(categorical, "16_categorical_quality.csv")

    cross = logical.copy()
    direct = df["거래유형"].str.strip().eq("직거래")
    broker_missing = missing_masks["중개사소재지"]
    direct_valid = direct & broker_missing
    if direct_valid.any():
        t = df.loc[direct_valid, ["__row_id", "__source_row_number", "중개사소재지"]].copy()
        t.columns = ["__row_id", "__source_row_number", "current_value"]
        t["issue_type"] = "STRUCTURAL_RULE"
        t["column"] = "중개사소재지"
        t["reason_flagged"] = "직거래이므로 중개사소재지 결측은 구조적으로 타당"
        t["status"] = "LIKELY_VALID"
        cross = pd.concat([cross, t[cross.columns]], ignore_index=True)
    cancellation_present = ~missing_masks["해제사유발생일"]
    if cancellation_present.any():
        t = df.loc[cancellation_present, ["__row_id", "__source_row_number", "해제사유발생일"]].copy()
        t.columns = ["__row_id", "__source_row_number", "current_value"]
        t["issue_type"] = "CANCELLATION_RECORD"
        t["column"] = "해제사유발생일"
        t["reason_flagged"] = "해제 거래 가능성: 가격 시계열/MDD 산출 포함 여부 결정 필요"
        t["status"] = "REVIEW_REQUIRED"
        cross = pd.concat([cross, t[cross.columns]], ignore_index=True)
    save_csv(cross, "17_cross_field_validation.csv")

    log("[7/10] 단지 조인키 품질")
    complex_df = pd.read_csv(COMPLEX, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["__join_key"] = normalize_text(df["시군구"]) + "|" + normalize_text(df["단지명"])
    complex_df["__join_key"] = normalize_text(complex_df["법정동주소"]) + "|" + normalize_text(complex_df["단지명"])
    complex_key_count = complex_df.groupby("__join_key").size()
    tx_key_count = df.groupby("__join_key").size()
    matched = df["__join_key"].isin(complex_key_count.index)
    duplicate_dim_keys = complex_key_count[complex_key_count.gt(1)].index
    expansion = df["__join_key"].isin(duplicate_dim_keys)
    join_quality = pd.DataFrame(
        [
            {"check": "거래키 NULL", "count": int(df["__join_key"].eq("|").sum()), "rate": float(df["__join_key"].eq("|").mean()), "status": "REVIEW_REQUIRED"},
            {"check": "단지키 중복", "count": int(complex_key_count.gt(1).sum()), "rate": float(complex_key_count.gt(1).mean()), "status": "REVIEW_REQUIRED"},
            {"check": "거래행 exact match", "count": int(matched.sum()), "rate": float(matched.mean()), "status": "LIKELY_VALID"},
            {"check": "거래행 unmatched", "count": int((~matched).sum()), "rate": float((~matched).mean()), "status": "REVIEW_REQUIRED"},
            {"check": "join explosion 위험 거래행", "count": int(expansion.sum()), "rate": float(expansion.mean()), "status": "REVIEW_REQUIRED"},
            {"check": "거래 unique key", "count": int(len(tx_key_count)), "rate": np.nan, "status": "LIKELY_VALID"},
        ]
    )
    save_csv(join_quality, "18_join_key_quality.csv")
    unmatched = df.loc[~matched, original_cols + ["__source_file", "__source_row_number", "__row_id", "__join_key"]].copy()
    unmatched.insert(0, "status", "REVIEW_REQUIRED")
    unmatched.insert(1, "reason", "법정동주소 정규화 + 단지명 정규화 exact match 실패")
    save_csv(unmatched, "19_unmatched_rows.csv")
    if expansion.any():
        expansion_rows = df.loc[expansion, original_cols + ["__source_file", "__source_row_number", "__row_id", "__join_key"]].merge(
            complex_df[["__join_key", "KAPT단지코드", "단지명", "법정동주소"]].rename(columns={"단지명": "단지_단지명", "법정동주소": "단지_법정동주소"}),
            on="__join_key",
            how="left",
        )
        expansion_rows.insert(0, "status", "REVIEW_REQUIRED")
    else:
        expansion_rows = pd.DataFrame(columns=["status", "__join_key", "__row_id", "KAPT단지코드"])
    save_csv(expansion_rows, "20_join_expansion_rows.csv")

    log("[8/10] 전처리 의사결정 로그")
    decisions: list[pd.DataFrame] = []

    def decision_from(frame: pd.DataFrame, issue_type: str, column: str, reason: str, recommendation: str) -> None:
        if len(frame) == 0 or "__row_id" not in frame.columns:
            return
        t = frame[["__row_id"]].drop_duplicates().copy()
        t["issue_type"] = issue_type
        t["column"] = column
        t["current_value"] = ""
        t["reason_flagged"] = reason
        t["ai_recommendation"] = recommendation
        decisions.append(t)

    decision_from(exact, "EXACT_DUPLICATE", "전체열", "원본 전체 열 값 동일", "원천 거래 ID 확보 후 중복 여부 확인")
    decision_from(business, "TRANSACTION_DUPLICATE_CANDIDATE", "거래복합키", "동일 거래조건의 복수 행", "자동 삭제 금지; 등기·거래 식별자와 대조")
    decision_from(miss_rows.loc[miss_rows["missing_semantic_type"].eq("POSSIBLE_DATA_QUALITY_ISSUE")], "MISSING", "복수", "비구조적 결측 포함", "주소/단지 마스터로 보완 가능성 검토")
    decision_from(logical.loc[logical["status"].ne("LIKELY_VALID")], "LOGICAL_OR_FORMAT", "복수", "논리·형식 검증 위반", "원천 공시 및 거래 상태 확인")
    decision_from(outlier_rows, "OUTLIER", "복수", "IQR/MAD/조건부 기준 이상치", "주변 거래와 비교 후 유지·제외 결정")
    decision_from(unmatched, "JOIN_UNMATCHED", "join_key", "단지 마스터 exact match 실패", "주소·단지명 표기 정규화 매핑 테이블 작성")
    decision_log = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame(columns=["__row_id", "issue_type", "column", "current_value", "reason_flagged", "ai_recommendation"])
    decision_log.insert(0, "issue_id", [f"ISSUE_{i:08d}" for i in range(1, len(decision_log) + 1)])
    for col in ["human_decision", "human_reason", "action", "completed"]:
        decision_log[col] = ""
    save_csv(decision_log, "preprocessing_decision_log.csv")

    log("[9/10] 시각화")
    setup_plotting()
    missing_plot = missing_summary.sort_values("missing_rate", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(missing_plot["column"], missing_plot["missing_rate"] * 100, color="#2563eb")
    plt.xlabel("결측률(%)")
    plt.title("원본 컬럼별 결측률")
    save_plot("01_missing_rate.png")

    price = df["__num_거래금액"].dropna()
    plt.figure(figsize=(9, 5)); plt.hist(price, bins=100, color="#0f766e"); plt.xlabel("거래금액(만원)"); plt.ylabel("건수"); plt.title("거래금액 분포"); save_plot("02_price_histogram.png")
    plt.figure(figsize=(9, 5)); plt.hist(np.log1p(price), bins=100, color="#0f766e"); plt.xlabel("log(1+거래금액)"); plt.ylabel("건수"); plt.title("로그 거래금액 분포"); save_plot("03_log_price_histogram.png")
    plt.figure(figsize=(9, 5)); plt.hist(df["__num_전용면적"].dropna(), bins=100, color="#7c3aed"); plt.xlabel("전용면적(㎡)"); plt.ylabel("건수"); plt.title("전용면적 분포"); save_plot("04_area_histogram.png")
    plt.figure(figsize=(9, 3)); plt.boxplot(price, orientation="horizontal", showfliers=False); plt.xlabel("거래금액(만원)"); plt.title("거래금액 박스플롯(극단점 표시는 생략)"); save_plot("05_price_boxplot.png")
    plt.figure(figsize=(9, 3)); plt.boxplot(df["__㎡당가격"].dropna(), orientation="horizontal", showfliers=False); plt.xlabel("㎡당가격(만원/㎡)"); plt.title("㎡당가격 박스플롯(극단점 표시는 생략)"); save_plot("06_unit_price_boxplot.png")

    district_data = [g["__㎡당가격"].dropna().values for _, g in df.groupby("__자치구") if len(g)]
    district_labels = [k for k, g in df.groupby("__자치구") if len(g)]
    plt.figure(figsize=(13, 6)); plt.boxplot(district_data, tick_labels=district_labels, showfliers=False); plt.xticks(rotation=60, ha="right"); plt.ylabel("㎡당가격(만원/㎡)"); plt.title("자치구별 ㎡당가격"); save_plot("07_district_unit_price_boxplot.png")
    area_data = [g["__㎡당가격"].dropna().values for _, g in df.groupby("__면적구", observed=True) if len(g)]
    area_labels_used = [str(k) for k, g in df.groupby("__면적구", observed=True) if len(g)]
    plt.figure(figsize=(10, 5)); plt.boxplot(area_data, tick_labels=area_labels_used, showfliers=False); plt.ylabel("㎡당가격(만원/㎡)"); plt.title("면적구별 ㎡당가격"); save_plot("08_area_unit_price_boxplot.png")

    monthly = df.dropna(subset=["__계약일자"]).set_index("__계약일자").resample("MS").agg(transaction_count=("__row_id", "size"), median_price=("__num_거래금액", "median"))
    plt.figure(figsize=(12, 5)); plt.plot(monthly.index, monthly["transaction_count"], color="#2563eb"); plt.ylabel("거래건수"); plt.title("계약월별 거래건수"); save_plot("09_monthly_transaction_count.png")
    plt.figure(figsize=(12, 5)); plt.plot(monthly.index, monthly["median_price"], color="#dc2626"); plt.ylabel("중앙 거래금액(만원)"); plt.title("계약월별 중앙 거래금액"); save_plot("10_monthly_median_price.png")
    plt.figure(figsize=(9, 5)); plt.hist(timeseries_quality["transaction_count"].clip(upper=timeseries_quality["transaction_count"].quantile(0.99)), bins=60, color="#d97706"); plt.xlabel("단지×면적구 거래건수(99% 절단 표시)"); plt.ylabel("그룹 수"); plt.title("단지×면적구별 거래건수 분포"); save_plot("11_group_transaction_count.png")
    plt.figure(figsize=(8, 7)); valid_coord = lat.notna() & lon.notna(); plt.scatter(lon[valid_coord], lat[valid_coord], s=6, alpha=0.35, color="#0f766e"); plt.xlabel("경도"); plt.ylabel("위도"); plt.title("입지 데이터 WGS84 좌표"); save_plot("12_coordinate_scatter.png")

    log("[10/10] 보고서 및 검증")
    after_hash = sha256(INPUT)
    exact_groups = exact["duplicate_group_id"].nunique() if len(exact) else 0
    business_groups = business["duplicate_candidate_group_id"].nunique() if len(business) else 0
    duplicate_key_affected_rows = int(duplicate_keys["count"].sum()) if len(duplicate_keys) else 0
    missing_nonstructural = int(miss_rows["missing_semantic_type"].eq("POSSIBLE_DATA_QUALITY_ISSUE").sum())
    build_after = int((df["__num_건축년도"] > df["__계약연도"]).sum())
    cancelled = int(cancellation_present.sum())
    low_reliability = int(timeseries_quality["mdd_reliability"].eq("LOW").sum())
    report = f"""# 아파트 실거래가 EDA 결과 보고서

> 분석 기준: `아파트_기본_DATA_raw/EDA_전처리_계획서_Codex.md`  
> 분석 원칙: 원천 파일 비수정, 자동 삭제·대체·보정 금지, 모든 검토 후보에 원본 추적키 부여  
> 실행 도구: Python(pandas, NumPy, Matplotlib)

## 1. 데이터 개요

- 주 분석 파일: `{SOURCE_NAME}`
- 데이터 크기: **{len(df):,}행 × {len(original_cols)}열**
- 계약일자 범위: **{df['__계약일자'].min().date()} ~ {df['__계약일자'].max().date()}**
- 자치구 수: **{df['__자치구'].replace('', pd.NA).nunique()}개**, 단지명 수: **{df['단지명'].nunique():,}개**
- 원천 추적키: `__source_file`, `__source_row_number`, `__row_id`를 분석용 데이터에만 생성
- 원본 SHA-256은 분석 전후 동일: **{'예' if before_hash == after_hash else '아니오'}**

## 2. 주요 품질 이슈 TOP 10

1. **구조적 결측이 광범위함**: 전체 {int(missing_any.sum()):,}행에 하나 이상의 결측 토큰이 있으나, 과거 공개 형식에서 `-`로 제공된 매수자·매도자·거래유형·등기일자 등이 대부분이므로 일괄 오류 처리하면 안 됩니다.
2. **비구조적 결측 후보**: {missing_nonstructural:,}행은 식별·분석 핵심 열 결측이 섞여 있어 별도 확인이 필요합니다.
3. **거래 복합키 잠재 중복**: {len(business):,}행 / {business_groups:,}그룹입니다. 동일 조건의 복수 정상 거래가 가능하므로 자동 삭제 금지입니다.
4. **전체 20열 완전 중복**: {len(exact):,}행 / {exact_groups:,}그룹입니다.
5. **해제 거래 후보**: 해제사유발생일이 있는 {cancelled:,}행은 MDD·회복률 계산 포함 여부를 확정해야 합니다.
6. **통계·조건부 이상치 후보**: 고유 {len(outlier_rows):,}행입니다. 이는 오류 확정값이 아니라 주변 거래 비교 대상입니다.
7. **건축년도 > 계약연도**: {build_after:,}행으로, 분양권·입주권/준공 전 거래 또는 원천 오류를 구분해야 합니다.
8. **시계열 이상 분기**: {int(q['timeseries_outlier'].sum()):,}개 단지×면적구×분기가 직전·rolling 기준에서 급변했습니다.
9. **MDD 신뢰도 낮은 그룹**: {low_reliability:,}개 단지×면적구는 관측 분기 또는 거래건수가 부족합니다.
10. **단지 exact join 미매칭**: {int((~matched).sum()):,}행({(~matched).mean():.2%})이며, 단지명/주소 표기 정규화·외부 매핑이 필요합니다.

## 3. 중복

- Level A(원본 20열 완전 동일): **{len(exact):,}행 / {exact_groups:,}그룹**
- Level B(`NO` 중복): **{duplicate_key_affected_rows:,}영향 행 / {len(duplicate_keys):,}중복 키 그룹**
- Level C(시군구·단지명·면적·계약일·금액·층·도로명): **{len(business):,}행 / {business_groups:,}그룹**
- Level D(공백·구두점 제거 후 유사 표기): **{len(near):,}개 정규화 그룹**
- 권고: 거래 고유 ID나 등기 식별자가 없으므로 중복 후보를 개수만으로 삭제하지 말고 `04_transaction_duplicate_candidates.csv`에서 원본 행을 검토하십시오.

## 4. 결측

- 결측 토큰(`NaN`, 빈 문자열, 공백, `-`, `NA`, `N/A`, `NULL`, `None`) 중 하나 이상을 가진 행: **{int(missing_any.sum()):,}행({missing_any.mean():.2%})**
- 구조적 결측만 가진 행: **{int(miss_rows['missing_semantic_type'].eq('STRUCTURAL_MISSING').sum()):,}행**
- 비구조적 결측 가능 행: **{missing_nonstructural:,}행**
- 결측률 상위 열:
{chr(10).join(f"  - {r.column}: {int(r.missing_count):,}건 ({r.missing_rate:.2%})" for r in missing_summary.sort_values('missing_rate', ascending=False).head(8).itertuples())}

## 5. 이상치

- 물리·논리 범위 검사는 면적/금액 0 이하, 유효하지 않은 계약일자, 건축년도-계약연도 역전, 층 숫자 변환·범위를 별도 점검했습니다.
- 전역 IQR과 MAD는 **삭제 기준이 아니라 후보 탐색 기준**으로만 사용했습니다.
- 가격은 `단지명 × 면적구 × 계약연도`의 ㎡당가격 MAD 기준을 추가 적용했습니다(n≥5).
- 전역·조건부 기준 중 하나 이상에 걸린 고유 행: **{len(outlier_rows):,}행({len(outlier_rows)/len(df):.2%})**
- 조건부 MAD 후보: **{int(group_mask.sum()):,}행**
- 주변 거래 파일은 조건부 MAD 후보마다 같은 단지·면적구의 시간상 최근 5건을 제공합니다.

## 6. 시계열 품질

- 단지×면적구 그룹: **{len(timeseries_quality):,}개**
- 시계열 급변 후보 분기: **{int(q['timeseries_outlier'].sum()):,}개**
- MDD 신뢰도 `LOW`: **{low_reliability:,}개 그룹**
- 고점/저점 원천 거래는 거래 8건 이상·관측 4분기 이상인 그룹만 추적했습니다. 고점·저점이 단 1건 거래로 결정되는지는 `13_timeseries_quality.csv`의 분기 건수와 `14_peak_trough_source_rows.csv`로 판단할 수 있습니다.
- 권고: 해제 거래와 조건부 이상치 1건이 분기 중앙값·고점·저점을 바꾸는지 민감도 분석 후 MDD/회복률을 확정하십시오.

## 7. 좌표

- 주 원천 거래 파일에는 좌표가 없어 별도 입지 테이블 `{LOCATION.name}`의 WGS84를 점검했습니다.
- 위도 37~38, 경도 126~128 범위·결측·X/Y 뒤바뀜 후보: **{len(coord_anomalies):,}행**
- GRS80TM 열은 현재 입지 테이블에 없어 단위·축 순서 검증 대상이 아닙니다.

## 8. Join Key 품질

- 기준: 거래 `시군구 + 단지명` ↔ 단지 `법정동주소 + 단지명`, 공백·구두점·대소문자 정규화 exact match
- 거래행 매칭: **{int(matched.sum()):,}행({matched.mean():.2%})**
- 미매칭: **{int((~matched).sum()):,}행({(~matched).mean():.2%})**
- 단지 차원 중복키: **{int(complex_key_count.gt(1).sum()):,}개**, join explosion 위험 거래행: **{int(expansion.sum()):,}행**
- exact match 실패는 곧 미존재를 뜻하지 않습니다. 법정동/도로명 주소와 단지 별칭을 이용한 승인형 매핑 테이블이 필요합니다.

## 9. MDD·회복률 분석에 미칠 영향

- 중복 후보를 모두 포함하면 특정 분기의 거래건수 가중이 커지고, 모두 제거하면 정상적인 동시 거래를 잃을 수 있습니다.
- 해제 거래는 실제 체결 시계열과 취소 시계열을 섞어 고점·저점을 왜곡할 수 있습니다.
- 거래 1건뿐인 분기와 장기 공백은 분기 중앙값을 불안정하게 하며 MDD/회복률의 시점을 이동시킬 수 있습니다.
- 고가 단지·대형 평형은 전역 IQR에서 정상임에도 이상치가 될 수 있으므로 조건부 기준과 주변 거래를 우선하십시오.
- 단지 조인 미매칭을 누락시키면 표본 선택 편향이 발생할 수 있습니다.

## 10. 전처리 후보와 사람의 결정이 필요한 항목

| 우선순위 | 항목 | 권고 상태 | 사람의 결정 |
|---:|---|---|---|
| 1 | 해제 거래 {cancelled:,}행 | REVIEW_REQUIRED | 시계열에서 제외/별도 시계열/유지 |
| 2 | 거래 복합키 잠재 중복 {business_groups:,}그룹 | REVIEW_REQUIRED | 거래 고유 ID 대조 후 KEEP/DROP |
| 3 | 단지 조인 미매칭 {int((~matched).sum()):,}행 | REVIEW_REQUIRED | 별칭·주소 매핑 승인 |
| 4 | 조건부 가격 이상치 {int(group_mask.sum()):,}행 | REVIEW_REQUIRED | 주변 5건과 비교 후 유지/제외 |
| 5 | 건축년도 역전 {build_after:,}행 | REVIEW_REQUIRED | 분양권·입주권 여부 확인 |
| 6 | 비구조적 결측 {missing_nonstructural:,}행 | REVIEW_REQUIRED | 보완/지표 제외/외부 확인 |
| 7 | MDD 신뢰도 LOW {low_reliability:,}그룹 | UNRESOLVED | 최소 거래·분기 기준 확정 |

`preprocessing_decision_log.csv`의 `human_decision`, `human_reason`, `action`, `completed` 열은 의도적으로 비워 두었습니다. 허용값은 `KEEP`, `DROP`, `CORRECT`, `IMPUTE`, `EXCLUDE_FROM_METRIC`, `NEED_EXTERNAL_CHECK` 중에서 사람이 확정해야 합니다.

## 11. 산출물 안내

- 핵심 보고서: `EDA_REPORT.md`
- 진단 표: `00_dataset_inventory.csv` ~ `20_join_expansion_rows.csv`
- 전처리 검토 로그: `preprocessing_decision_log.csv`
- 시각화: `figures/01_missing_rate.png` ~ `figures/12_coordinate_scatter.png`
- 재현 코드: `03_단지매핑_입지결합/eda_보고서/run_eda.py`

## 12. 결론

원천 데이터는 가격·면적·날짜의 기본 숫자 변환은 가능하지만, **해제 거래, 잠재 중복, 단지 조인 미매칭, 희소 분기, 조건부 가격 이상치**를 먼저 사람이 결정해야 안정적인 MDD·회복률 분석이 가능합니다. 본 EDA에서는 후보를 삭제하거나 값을 보정하지 않았습니다.
"""
    (OUTPUT / "EDA_REPORT.md").write_text(report, encoding="utf-8")
    result = {
        "input": str(INPUT),
        "rows": len(df),
        "columns": len(original_cols),
        "date_min": str(df["__계약일자"].min().date()),
        "date_max": str(df["__계약일자"].max().date()),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "source_unchanged": before_hash == after_hash,
        "exact_duplicate_rows": len(exact),
        "duplicate_key_affected_rows": duplicate_key_affected_rows,
        "duplicate_key_groups": len(duplicate_keys),
        "transaction_duplicate_candidate_rows": len(business),
        "rows_with_missing": int(missing_any.sum()),
        "nonstructural_missing_rows": missing_nonstructural,
        "outlier_candidate_rows": len(outlier_rows),
        "timeseries_outlier_quarters": int(q["timeseries_outlier"].sum()),
        "coordinate_anomaly_rows": len(coord_anomalies),
        "join_unmatched_rows": int((~matched).sum()),
        "decision_log_rows": len(decision_log),
    }
    (OUTPUT / "EDA_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
