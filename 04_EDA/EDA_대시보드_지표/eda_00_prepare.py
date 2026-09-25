"""1단계: 대시보드가 읽는 CSV를 housing.data 파이프라인 그대로 통과시켜 EDA용 중간 산출물을 만든다.

원본 CSV(883MB)를 매번 다시 읽지 않도록 다음 parquet을 캐시에 저장한다.
  trades.parquet     거래 단위 유효 표본 (대시보드 집계의 원재료)
  quarterly.parquet  단지 x 면적군 x 분기 집계 (대시보드가 실제로 쓰는 시계열)
  complexes.parquet  단지 속성
  facilities.parquet 단지별 최근접 생활시설
  meta.json          진단 지표 / 기간 / 기준 분기

실행: python 04_EDA/EDA_대시보드_지표/eda_00_prepare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "대시보드"))

from housing.data import DEFAULT_CSV, READ_COLUMNS, area_groups, numeric, prepare_data

CACHE = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent / "_cache")
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV


def derive_trades(raw: pd.DataFrame) -> pd.DataFrame:
    """housing.data.prepare_data와 동일한 유효성 규칙으로 거래 단위 표본을 재현한다.

    prepare_data는 집계 결과만 돌려주므로, 가격·면적 분포 EDA를 위해 거래 단위 프레임이 따로 필요하다.
    유효 건수를 파이프라인 진단값과 대조해 로직이 어긋나면 즉시 실패시킨다.
    """
    df = raw.reindex(columns=READ_COLUMNS).fillna("")
    for col in READ_COLUMNS:
        df[col] = df[col].astype(str).str.strip().replace("-", "")

    month = df["계약년월"].str.replace("-", "", regex=False)
    day = numeric(df["계약일"])
    integer_day = day.where(day.eq(day.round()) & day.between(1, 31)).astype("Int64")
    df["date"] = pd.to_datetime(month + integer_day.astype("string").str.zfill(2),
                               format="%Y%m%d", errors="coerce")
    df["area"] = numeric(df["전용면적(㎡)"])
    df["amount"] = numeric(df["거래금액(만원)"])

    cancelled = df["해제사유발생일"].ne("")
    bad_date = df["date"].isna()
    bad_price = ~np.isfinite(df["amount"]) | df["amount"].le(0)
    bad_area = ~np.isfinite(df["area"]) | df["area"].le(0)
    bad_identity = df["단지명"].eq("") | df["시군구"].eq("")
    valid = ~(cancelled | bad_date | bad_price | bad_area | bad_identity)

    exclusion = pd.DataFrame({
        "해제 거래": cancelled, "계약일 오류": bad_date, "가격 오류": bad_price,
        "면적 오류": bad_area, "단지명·지역 누락": bad_identity,
    })

    out = df.loc[valid, ["시군구", "단지명", "date", "area", "amount",
                         "단지매칭상태", "단지_KAPT단지코드", "건축년도"]].copy()
    place = out["시군구"].str.split()
    out["district"] = place.str[1].where(place.str.len() > 1, place.str[0])
    out["dong"] = place.str[2:].str.join(" ")
    out["area_group"] = area_groups(out["area"]).astype(str)
    out["quarter"] = out["date"].dt.to_period("Q")
    out["unit_price"] = out["amount"] / out["area"]
    out["matched"] = out["단지매칭상태"].eq("매칭완료") & out["단지_KAPT단지코드"].ne("")
    out["built_year"] = numeric(out["건축년도"])
    return out.drop(columns=["단지매칭상태", "단지_KAPT단지코드", "건축년도"]), exclusion.loc[~valid]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"CSV 로딩: {CSV}")
    raw = pd.read_csv(CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False,
                      usecols=lambda c: c in set(READ_COLUMNS))
    print(f"  읽은 행/열: {raw.shape}")

    data = prepare_data(raw, CSV.name)
    print(f"  유효 거래 {data.diagnostics['유효 거래']:,} / 고유 단지 {data.diagnostics['고유 단지']:,}")

    trades, excluded = derive_trades(raw)
    assert len(trades) == data.diagnostics["유효 거래"], (
        f"거래 단위 재현이 파이프라인과 어긋남: {len(trades)} != {data.diagnostics['유효 거래']}")
    del raw

    # 제외 사유는 서로 겹칠 수 있으므로 사유별 합계와 실제 제외 행 수를 같이 남긴다.
    reasons = {col: int(excluded[col].sum()) for col in excluded.columns}
    reasons["제외 행 수(중복 제거)"] = len(excluded)

    trades["quarter"] = trades["quarter"].astype(str)
    trades.to_parquet(CACHE / "trades.parquet", index=False)
    q = data.quarterly.copy()
    q["quarter"] = q["quarter"].astype(str)
    q.to_parquet(CACHE / "quarterly.parquet", index=False)
    data.complexes.to_parquet(CACHE / "complexes.parquet", index=False)
    data.facilities.to_parquet(CACHE / "facilities.parquet", index=False)

    meta = {
        "csv": str(CSV), "csv_mb": round(CSV.stat().st_size / 2**20, 1),
        "start": str(data.start.date()), "end": str(data.end.date()),
        "reference": str(data.reference), "partial": bool(data.partial),
        "diagnostics": data.diagnostics, "exclusion_reasons": reasons,
    }
    (CACHE / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n캐시 저장 완료 -> {CACHE}")


if __name__ == "__main__":
    main()
