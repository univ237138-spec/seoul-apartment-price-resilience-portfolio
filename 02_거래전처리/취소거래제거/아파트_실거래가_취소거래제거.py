"""해제사유발생일이 있는 취소 거래를 제외한 새 CSV와 보고서를 생성한다."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_CSV = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용.csv"
OUTPUT_CSV = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_취소거래제거.csv"
REPORT_MD = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_취소거래제거_분석결과.md"
TEMP_CSV = BASE_DIR / ".아파트_실거래가_취소거래제거.tmp.csv"
CHUNK_SIZE = 100_000
CANCELLATION_COLUMN = "해제사유발생일"
DATE_FORMAT = "%Y-%m-%d"


def sha256_file(path: Path) -> str:
    """파일의 SHA-256 해시를 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def human_bytes(size: int) -> str:
    """바이트 수를 읽기 쉬운 이진 단위로 표시한다."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.2f} {unit}"
        value /= 1024
    raise AssertionError("도달할 수 없는 코드")


def cancellation_dates(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """취소 여부와 파싱된 취소일을 반환하고 잘못된 날짜값을 차단한다."""
    normalized = series.astype("string").str.strip()
    has_cancellation_date = normalized.notna() & ~normalized.isin(["", "-"])
    parsed = pd.to_datetime(normalized.where(has_cancellation_date), format=DATE_FORMAT, errors="coerce")

    invalid = has_cancellation_date & parsed.isna()
    if invalid.any():
        examples = normalized[invalid].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{CANCELLATION_COLUMN}에 {DATE_FORMAT} 형식이 아닌 값이 있습니다: {examples}"
        )
    return has_cancellation_date, parsed


def validate_output(expected_columns: list[str], expected_rows: int) -> int:
    """출력 CSV의 행·컬럼과 취소 거래 잔존 여부를 전체 검사한다."""
    checked_rows = 0
    remaining_cancellations = 0
    for chunk in pd.read_csv(
        TEMP_CSV,
        encoding="utf-8-sig",
        dtype="string",
        chunksize=CHUNK_SIZE,
    ):
        if chunk.columns.tolist() != expected_columns:
            raise ValueError("출력 CSV의 컬럼 수 또는 순서가 원본과 다릅니다.")
        cancelled, _ = cancellation_dates(chunk[CANCELLATION_COLUMN])
        remaining_cancellations += int(cancelled.sum())
        checked_rows += len(chunk)

    if checked_rows != expected_rows:
        raise ValueError(
            f"출력 행 수 검증 실패: 예상 {expected_rows:,}행, 실제 {checked_rows:,}행"
        )
    if remaining_cancellations:
        raise ValueError(f"출력 CSV에 취소 거래 {remaining_cancellations:,}건이 남아 있습니다.")
    return checked_rows


def write_report(
    *,
    source_hash: str,
    output_hash: str,
    source_rows: int,
    removed_rows: int,
    output_rows: int,
    expected_columns: list[str],
    cancel_min: pd.Timestamp,
    cancel_max: pd.Timestamp,
    cancellation_years: Counter[int],
) -> None:
    """전처리 기준, 집계 및 검증 결과를 Markdown으로 기록한다."""
    removal_rate = removed_rows / source_rows if source_rows else 0.0
    lines = [
        "# 아파트(매매) 실거래가 취소 거래 제거 결과",
        "",
        f"- 작업 시각: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 원본 파일: `{SOURCE_CSV.name}`",
        f"- 전처리 파일: `{OUTPUT_CSV.name}`",
        f"- 원본 데이터: **{source_rows:,}행 × {len(expected_columns)}열**",
        f"- 제거된 취소 거래: **{removed_rows:,}행 ({removal_rate:.2%})**",
        f"- 전처리 후 데이터: **{output_rows:,}행 × {len(expected_columns)}열**",
        f"- 원본 파일 크기: **{SOURCE_CSV.stat().st_size:,} bytes ({human_bytes(SOURCE_CSV.stat().st_size)})**",
        f"- 전처리 파일 크기: **{OUTPUT_CSV.stat().st_size:,} bytes ({human_bytes(OUTPUT_CSV.stat().st_size)})**",
        "",
        "## 전처리 기준",
        "",
        f"- 판정 컬럼: `{CANCELLATION_COLUMN}`",
        "- 취소 거래: 해당 컬럼에 `YYYY-MM-DD` 형식의 실제 날짜가 있는 행",
        "- 유지 거래: 해당 컬럼이 빈 값, 결측값 또는 `-`인 행",
        "- 처리 방법: 취소 거래로 판정된 행 전체를 새 CSV에서 제외",
        "- 다른 컬럼의 값과 컬럼 순서는 변경하지 않음",
        "",
        "## 제거된 취소 거래 요약",
        "",
        f"- 해제사유발생일 최소값: **{cancel_min.strftime('%Y-%m-%d')}**",
        f"- 해제사유발생일 최대값: **{cancel_max.strftime('%Y-%m-%d')}**",
        "",
        "| 해제 연도 | 제거 건수 |",
        "|---:|---:|",
    ]
    for year in sorted(cancellation_years):
        lines.append(f"| {year} | {cancellation_years[year]:,} |")

    lines.extend(
        [
            "",
            "## 검증 결과",
            "",
            f"- 행 수 산식: **{source_rows:,} - {removed_rows:,} = {output_rows:,} (일치)**",
            f"- 컬럼 수 및 순서 유지: **통과({len(expected_columns)}열)**",
            f"- 전처리 후 `{CANCELLATION_COLUMN}`에 날짜가 남은 행: **0건**",
            "- 취소일 형식 오류: **0건**",
            "- 원본 파일 미변경: **통과(SHA-256 작업 전후 동일)**",
            f"- 원본 SHA-256: `{source_hash}`",
            f"- 전처리 파일 SHA-256: `{output_hash}`",
            "",
            "## 재현 방법",
            "",
            "같은 폴더에서 아래 명령을 실행하면 동일한 전처리를 다시 수행합니다.",
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

    expected_columns = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig", nrows=0).columns.tolist()
    if CANCELLATION_COLUMN not in expected_columns:
        raise KeyError(f"필수 컬럼이 없습니다: {CANCELLATION_COLUMN}")

    source_hash_before = sha256_file(SOURCE_CSV)
    source_rows = 0
    removed_rows = 0
    output_rows = 0
    cancel_min: pd.Timestamp | None = None
    cancel_max: pd.Timestamp | None = None
    cancellation_years: Counter[int] = Counter()

    try:
        with TEMP_CSV.open("w", encoding="utf-8-sig", newline="") as output_file:
            first_chunk = True
            for chunk in pd.read_csv(
                SOURCE_CSV,
                encoding="utf-8-sig",
                dtype="string",
                chunksize=CHUNK_SIZE,
            ):
                if chunk.columns.tolist() != expected_columns:
                    raise ValueError("처리 중 원본 CSV의 컬럼 구성이 달라졌습니다.")

                cancelled, parsed_dates = cancellation_dates(chunk[CANCELLATION_COLUMN])
                kept = chunk.loc[~cancelled]
                cancelled_dates = parsed_dates[cancelled]

                chunk_removed = int(cancelled.sum())
                source_rows += len(chunk)
                removed_rows += chunk_removed
                output_rows += len(kept)

                if chunk_removed:
                    current_min = cancelled_dates.min()
                    current_max = cancelled_dates.max()
                    cancel_min = current_min if cancel_min is None else min(cancel_min, current_min)
                    cancel_max = current_max if cancel_max is None else max(cancel_max, current_max)
                    cancellation_years.update(
                        int(year) for year in cancelled_dates.dt.year.dropna().astype(int)
                    )

                kept.to_csv(output_file, index=False, header=first_chunk)
                first_chunk = False

        if source_rows != removed_rows + output_rows:
            raise ValueError("원본 행 수가 제거 행 수와 유지 행 수의 합과 일치하지 않습니다.")
        if removed_rows == 0 or cancel_min is None or cancel_max is None:
            raise ValueError("제거할 취소 거래가 발견되지 않았습니다.")

        validate_output(expected_columns, output_rows)

        source_hash_after = sha256_file(SOURCE_CSV)
        if source_hash_before != source_hash_after:
            raise RuntimeError("전처리 도중 원본 파일이 변경되었습니다.")

        TEMP_CSV.replace(OUTPUT_CSV)
        output_hash = sha256_file(OUTPUT_CSV)
        write_report(
            source_hash=source_hash_before,
            output_hash=output_hash,
            source_rows=source_rows,
            removed_rows=removed_rows,
            output_rows=output_rows,
            expected_columns=expected_columns,
            cancel_min=cancel_min,
            cancel_max=cancel_max,
            cancellation_years=cancellation_years,
        )
    finally:
        TEMP_CSV.unlink(missing_ok=True)

    print(f"원본: {source_rows:,}행")
    print(f"취소 거래 제거: {removed_rows:,}행")
    print(f"전처리 완료: {output_rows:,}행 × {len(expected_columns)}열")
    print(f"새 CSV: {OUTPUT_CSV}")
    print(f"보고서: {REPORT_MD}")


if __name__ == "__main__":
    main()
