"""Read the designated CSV without modifying it or joining other files."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import AREA_BINS, AREA_LABELS, Dataset, FACILITY_TYPES
from .scoring import build_scores


DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "거래_단지_DATA"
    / "아파트_거래_DATA_대시보드_실데이터.csv.gz"
)
PIPELINE_VERSION = "2.1"
REQUIRED = ["시군구", "단지명", "전용면적(㎡)", "계약년월", "계약일", "거래금액(만원)"]
INFO_COLUMNS = [
    "단지명", "시군구", "도로명주소", "번지", "단지_KAPT단지코드",
    "단지_단지명", "단지매칭상태", "건축년도", "단지_세대수",
    "단지_지상주차대수", "단지_지하주차대수", "단지_복도유형", "위도", "경도", "입지매핑상태",
]
FACILITY_COLUMNS = [
    f"{kind}_{suffix}" for kind in FACILITY_TYPES
    for suffix in ["시설명", "세부유형", "주소", "위도", "경도", "거리_m", "입지원본행번호"]
]
# Every column the pipeline reads. Columns outside this list are left in the source file untouched.
READ_COLUMNS = list(dict.fromkeys(REQUIRED + ["해제사유발생일"] + INFO_COLUMNS + FACILITY_COLUMNS))


class DataError(ValueError):
    """An actionable error that may be safely shown in the app."""


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def number(value):
    try:
        n = float(str(value).replace(",", ""))
        return n if np.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def valid_coordinates(lat, lon) -> bool:
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def area_groups(values: pd.Series) -> pd.Series:
    return pd.cut(values, AREA_BINS, labels=AREA_LABELS, right=False)


def load_csv(path: str | Path = DEFAULT_CSV) -> Dataset:
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataError(f"CSV를 읽을 수 없습니다. 파일 위치와 UTF-8 인코딩을 확인해 주세요. ({type(exc).__name__})") from exc
    return prepare_data(raw, Path(path).name)


def prepare_data(raw: pd.DataFrame, filename: str = "CSV") -> Dataset:
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise DataError("필수 컬럼이 없습니다: " + ", ".join(missing))
    if raw.empty:
        raise DataError("CSV에 거래 행이 없습니다.")
    df = raw.reindex(columns=READ_COLUMNS).fillna("")
    for col in READ_COLUMNS:
        df[col] = df[col].astype(str).str.strip().replace("-", "")

    month = df["계약년월"].str.replace("-", "", regex=False)
    day = numeric(df["계약일"])
    integer_day = day.where(day.eq(day.round()) & day.between(1, 31)).astype("Int64")
    df["date"] = pd.to_datetime(month + integer_day.astype("string").str.zfill(2), format="%Y%m%d", errors="coerce")
    df["area"] = numeric(df["전용면적(㎡)"])
    df["amount"] = numeric(df["거래금액(만원)"])
    cancelled = df["해제사유발생일"].ne("")
    bad_date = df["date"].isna()
    bad_price = ~np.isfinite(df["amount"]) | df["amount"].le(0)
    bad_area = ~np.isfinite(df["area"]) | df["area"].le(0)
    bad_identity = df["단지명"].eq("") | df["시군구"].eq("")
    valid = ~(cancelled | bad_date | bad_price | bad_area | bad_identity)
    diagnostics = {
        "파일": filename, "원본 행": len(raw), "원본 컬럼": len(raw.columns),
        "유효 거래": int(valid.sum()), "제외 거래": int((~valid).sum()),
        "해제 거래": int(cancelled.sum()), "계약일 오류": int(bad_date.sum()),
        "가격 오류": int(bad_price.sum()), "면적 오류": int(bad_area.sum()),
        "단지명·지역 누락": int(bad_identity.sum()),
        "KAPT 코드 결측": int(df["단지_KAPT단지코드"].eq("").sum()),
    }
    df = df.loc[valid].copy()
    if df.empty:
        raise DataError("가격·면적·계약일 조건을 만족하는 유효 거래가 없습니다.")

    matched = df["단지매칭상태"].eq("매칭완료") & df["단지_KAPT단지코드"].ne("")
    parts = pd.DataFrame({"region": df["시군구"],
                          "place": df["도로명주소"].where(df["도로명주소"].ne(""), df["번지"]),
                          "name": df["단지명"]})
    # Hash once per distinct address instead of once per trade; json.dumps keeps fields
    # unambiguous, so a value containing a separator cannot merge two different complexes.
    distinct = parts.loc[~matched].drop_duplicates()
    digests = pd.Series(
        ["address:" + sha256(json.dumps(list(values), ensure_ascii=False).encode()).hexdigest()
         for values in distinct.itertuples(index=False, name=None)],
        index=pd.MultiIndex.from_frame(distinct))
    df["complex_id"] = "kapt:" + df["단지_KAPT단지코드"]
    df.loc[~matched, "complex_id"] = digests.reindex(
        pd.MultiIndex.from_frame(parts.loc[~matched])).to_numpy()
    df["area_group"] = area_groups(df["area"])
    df["quarter"] = df["date"].dt.to_period("Q")
    df["unit_price"] = df["amount"] / df["area"]
    start, end = df["date"].min(), df["date"].max()
    latest = end.to_period("Q")
    # The first partially covered quarter and the last observed quarter are not complete.
    quarters = pd.period_range(start.to_period("Q"), latest, freq="Q")
    complete = [q for q in quarters if q < latest and q.start_time >= start.normalize()]
    reference = latest
    quarterly = df.groupby(["complex_id", "area_group", "quarter"], observed=True, sort=True).agg(
        price=("unit_price", "median"), count=("unit_price", "size")
    ).reset_index()
    quarterly["area_group"] = quarterly["area_group"].astype(str)
    quarterly["complete"] = quarterly["quarter"].isin(complete)
    representative_areas = df.groupby(["complex_id", "area_group"], observed=True)["area"].median().to_dict()
    # 분기 지표를 못 내는 조합에도 "확인된 마지막 거래" 한 건은 보여줄 수 있게, 집계로 지우기 전에
    # 조합별 최근 1건만 남긴다. 분기 중위값이 아니라 실제로 체결된 거래다.
    last_trades = (df.sort_values("date").groupby(["complex_id", "area_group"], observed=True).tail(1)
                   .set_index(["complex_id", "area_group"])[["date", "amount", "area", "unit_price"]]
                   .to_dict("index"))

    records, facilities = [], []
    info_cols = INFO_COLUMNS + FACILITY_COLUMNS
    bundle_cols = ["complex_id"] + info_cols
    # A complex carrying a single distinct bundle has nothing to choose, so only the
    # complexes with disagreeing rows are counted bundle by bundle.
    distinct_bundles = df[bundle_cols].drop_duplicates()
    per_complex = distinct_bundles.groupby("complex_id", sort=False).size()
    conflicted = set(per_complex.index[per_complex.gt(1)])
    winners = distinct_bundles.loc[~distinct_bundles["complex_id"].isin(conflicted)]
    if conflicted:
        resolved = []
        ambiguous = df.loc[df["complex_id"].isin(conflicted), bundle_cols]
        for complex_id, group in ambiguous.groupby("complex_id", sort=False):
            counts = Counter(group[info_cols].itertuples(index=False, name=None))
            winner = max(counts, key=counts.get)  # Counter insertion order breaks ties by source row.
            resolved.append((complex_id, *winner))
        winners = pd.concat([winners, pd.DataFrame(resolved, columns=bundle_cols)])
    # Emit complexes in first-appearance order, as the per-complex loop used to.
    winners = winners.set_index("complex_id").loc[df["complex_id"].drop_duplicates()]
    for complex_id, bundle in zip(winners.index, winners.itertuples(index=False, name=None)):
        row = dict(zip(info_cols, bundle))
        conflict = complex_id in conflicted
        place = row["시군구"].split()
        lat, lon = number(row["위도"]), number(row["경도"])
        has_coords = valid_coordinates(lat, lon)
        records.append({
            "complex_id": complex_id,
            "name": row["단지명"], "registered_name": row["단지_단지명"],
            "district": place[1] if len(place) > 1 else place[0],
            "dong": " ".join(place[2:]),
            "address": row["도로명주소"] or f'{row["시군구"]} {row["번지"]}',
            "latitude": lat if has_coords else None, "longitude": lon if has_coords else None,
            "households": number(row["단지_세대수"]), "built_year": number(row["건축년도"]),
            "parking_ground": number(row["단지_지상주차대수"]),
            "parking_underground": number(row["단지_지하주차대수"]),
            "corridor_type": row["단지_복도유형"] or None,
            "bundle_conflict": conflict,
        })
        for kind in FACILITY_TYPES:
            dist = number(row[f"{kind}_거리_m"])
            flat, flon = number(row[f"{kind}_위도"]), number(row[f"{kind}_경도"])
            name = row[f"{kind}_시설명"]
            state = "no_coordinates" if not has_coords else "unmapped"
            if has_coords and name:
                state = "ok" if dist is not None and 0 <= dist <= 1000 else "invalid"
            facilities.append({
                "complex_id": complex_id, "kind": kind, "name": name,
                "subtype": row[f"{kind}_세부유형"], "distance": dist,
                "latitude": flat, "longitude": flon, "status": state,
                "map_valid": state == "ok" and valid_coordinates(flat, flon),
                "count": None, "count_radius": None,
            })
    complexes = pd.DataFrame(records)
    diagnostics["고유 단지"] = len(complexes)
    diagnostics["좌표 없는 단지"] = int(complexes["latitude"].isna().sum())
    diagnostics["단지·입지 정보 묶음 차이"] = len(conflicted)
    # Grades rank each combination against its area group, so they need the whole table at once.
    scores = build_scores(quarterly, reference, True)
    diagnostics["등급 산출 조합"] = len(scores)
    return Dataset("real", quarterly, complexes, pd.DataFrame(facilities), start, end,
                   reference, True, diagnostics, scores=scores,
                   representative_areas=representative_areas, last_trades=last_trades)
