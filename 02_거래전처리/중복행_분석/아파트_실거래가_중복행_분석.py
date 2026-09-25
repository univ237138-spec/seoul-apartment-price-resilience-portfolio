"""아파트 실거래가 CSV의 중복을 여러 기준으로 분석하고 해당 행을 추출한다."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_CSV = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_취소거래제거.csv"
REPORT_MD = BASE_DIR / "아파트(매매)_실거래가_2017_2026_시계열통합_권장타입적용_취소거래제거_중복분석결과.md"

OUTPUT_EXACT = BASE_DIR / "중복상황1_전체컬럼_완전동일행.csv"
OUTPUT_EXCEPT_NO = BASE_DIR / "중복상황2_NO제외_동일행.csv"
OUTPUT_CORE = BASE_DIR / "중복상황3_핵심거래키_동일행.csv"
OUTPUT_CORE_DIFFERENT = BASE_DIR / "중복상황4_핵심거래키동일_세부정보차이행.csv"
OUTPUT_NO = BASE_DIR / "중복상황5_NO_반복행.csv"

METADATA_COLUMNS = ["_중복그룹ID", "_그룹행수", "_원본CSV행번호"]
WRITE_CHUNK_SIZE = 100_000


@dataclass
class ScenarioResult:
    name: str
    interpretation: str
    keys: list[str]
    output_path: Path
    row_count: int
    group_count: int
    excess_count: int
    max_group_size: int
    sha256: str


def sha256_file(path: Path) -> str:
    """파일의 SHA-256 해시를 스트리밍 방식으로 계산한다."""
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


def duplicated_mask(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    """지정 키가 두 번 이상 등장하는 모든 행을 표시한다."""
    return frame.duplicated(subset=keys, keep=False)


def ambiguous_core_mask(
    frame: pd.DataFrame,
    core_mask: pd.Series,
    core_keys: list[str],
    content_keys: list[str],
) -> pd.Series:
    """핵심키는 같지만 NO 이외의 세부 내용이 다른 그룹의 모든 행을 표시한다."""
    core_rows = frame.loc[core_mask]
    core_group = core_rows.groupby(core_keys, dropna=False, sort=False).ngroup()
    content_group = core_rows.groupby(content_keys, dropna=False, sort=False).ngroup()
    pairs = pd.DataFrame({"core_group": core_group, "content_group": content_group})
    distinct_content = (
        pairs.drop_duplicates()
        .groupby("core_group", sort=False)["content_group"]
        .nunique()
    )
    ambiguous_groups = distinct_content[distinct_content.gt(1)].index
    selected_index = core_rows.index[core_group.isin(ambiguous_groups)]
    return pd.Series(frame.index.isin(selected_index), index=frame.index)


def extract_scenario(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    name: str,
    interpretation: str,
    keys: list[str],
    output_path: Path,
) -> ScenarioResult:
    """중복 그룹의 모든 원본 행에 검토용 메타데이터를 붙여 CSV로 저장한다."""
    selected_index = frame.index[mask].to_numpy(dtype=np.int64)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        if len(selected_index) == 0:
            pd.DataFrame(columns=METADATA_COLUMNS + frame.columns.tolist()).to_csv(
                temp_path, index=False, encoding="utf-8-sig"
            )
            group_count = 0
            excess_count = 0
            max_group_size = 0
        else:
            key_rows = frame.loc[selected_index, keys]
            group_codes = (
                key_rows.groupby(keys, dropna=False, sort=False).ngroup().to_numpy(dtype=np.int64)
                + 1
            )
            group_sizes_by_code = np.bincount(group_codes)
            row_group_sizes = group_sizes_by_code[group_codes]
            if np.any(row_group_sizes < 2):
                raise ValueError(f"{name}: 중복 추출에 단일 행 그룹이 포함되었습니다.")

            # 같은 그룹이 파일에서 연속하도록 그룹ID, 원본 행번호 순으로 정렬한다.
            order = np.lexsort((selected_index, group_codes))
            selected_index = selected_index[order]
            group_codes = group_codes[order]
            row_group_sizes = row_group_sizes[order]

            group_count = int(group_codes.max())
            excess_count = int(len(selected_index) - group_count)
            max_group_size = int(row_group_sizes.max())

            with temp_path.open("w", encoding="utf-8-sig", newline="") as output_file:
                first_chunk = True
                for start in range(0, len(selected_index), WRITE_CHUNK_SIZE):
                    end = start + WRITE_CHUNK_SIZE
                    batch_index = selected_index[start:end]
                    batch = frame.loc[batch_index].reset_index(drop=True)
                    metadata = pd.DataFrame(
                        {
                            "_중복그룹ID": group_codes[start:end],
                            "_그룹행수": row_group_sizes[start:end],
                            # 헤더가 1행이므로 pandas index 0은 원본 CSV의 2행이다.
                            "_원본CSV행번호": batch_index + 2,
                        }
                    )
                    exported = pd.concat([metadata, batch], axis=1)
                    exported.to_csv(output_file, index=False, header=first_chunk)
                    first_chunk = False

        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return ScenarioResult(
        name=name,
        interpretation=interpretation,
        keys=keys,
        output_path=output_path,
        row_count=int(len(selected_index)),
        group_count=group_count,
        excess_count=excess_count,
        max_group_size=max_group_size,
        sha256=sha256_file(output_path),
    )


def validate_extract(result: ScenarioResult, source_columns: list[str]) -> None:
    """추출 CSV의 행 수, 컬럼, 그룹 크기 메타데이터를 재검증한다."""
    expected_columns = METADATA_COLUMNS + source_columns
    actual_columns = pd.read_csv(
        result.output_path, encoding="utf-8-sig", nrows=0
    ).columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(f"{result.name}: 추출 CSV의 컬럼 구성이 올바르지 않습니다.")

    checked_rows = 0
    for chunk in pd.read_csv(
        result.output_path,
        encoding="utf-8-sig",
        dtype="string",
        chunksize=WRITE_CHUNK_SIZE,
    ):
        if len(chunk) and pd.to_numeric(chunk["_그룹행수"], errors="coerce").lt(2).any():
            raise ValueError(f"{result.name}: 잘못된 그룹행수 값이 있습니다.")
        checked_rows += len(chunk)
    if checked_rows != result.row_count:
        raise ValueError(
            f"{result.name}: 추출 행 수 불일치({result.row_count:,} != {checked_rows:,})"
        )


def varying_columns_by_core_group(
    frame: pd.DataFrame,
    mask: pd.Series,
    core_keys: list[str],
    content_keys: list[str],
) -> dict[str, int]:
    """핵심키가 같은 검토 그룹에서 각 세부 컬럼이 달라진 그룹 수를 센다."""
    rows = frame.loc[mask].copy()
    rows["_core_group"] = rows.groupby(core_keys, dropna=False, sort=False).ngroup()
    non_core_columns = [column for column in content_keys if column not in core_keys]
    variation_counts: dict[str, int] = {}
    for column in non_core_columns:
        counts = rows.groupby("_core_group", sort=False)[column].nunique(dropna=False)
        variation_counts[column] = int(counts.gt(1).sum())
    return variation_counts


def write_report(
    *,
    frame: pd.DataFrame,
    source_hash: str,
    results: list[ScenarioResult],
    core_keys: list[str],
    variations: dict[str, int],
) -> None:
    """중복 분석 기준, 수치, 해석, 추출 파일 정보를 Markdown으로 저장한다."""
    result_by_name = {result.name: result for result in results}
    exact = result_by_name["전체 컬럼 완전 동일"]
    except_no = result_by_name["NO 제외 동일"]
    core = result_by_name["핵심 거래키 동일"]
    ambiguous = result_by_name["핵심키 동일·세부정보 차이"]
    no_repeat = result_by_name["NO 반복"]

    lines = [
        "# 아파트(매매) 실거래가 중복행 분석 결과",
        "",
        f"- 분석 시각: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 원본 파일: `{SOURCE_CSV.name}`",
        f"- 데이터 크기: **{len(frame):,}행 × {len(frame.columns)}열**",
        f"- 원본 파일 크기: **{SOURCE_CSV.stat().st_size:,} bytes ({human_bytes(SOURCE_CSV.stat().st_size)})**",
        f"- 원본 SHA-256: `{source_hash}`",
        "- 분석 중 원본 파일 변경: **없음**",
        "",
        "## 핵심 결론",
        "",
        f"- 전체 20개 컬럼이 완전히 동일한 확정 중복행은 **{exact.row_count:,}행**입니다.",
        f"- `NO`만 제외하면 내용이 동일한 강한 중복 후보는 **{except_no.row_count:,}행, "
        f"{except_no.group_count:,}그룹**입니다. 중복 초과분은 {except_no.excess_count:,}행입니다.",
        f"- 핵심 거래정보가 같은 후보는 **{core.row_count:,}행, {core.group_count:,}그룹**입니다. "
        "동일 날짜에 같은 조건의 서로 다른 세대가 거래됐을 수도 있어 자동 삭제 대상으로 보면 안 됩니다.",
        f"- 핵심키는 같지만 NO 이외 세부정보가 다른 경우는 **{ambiguous.row_count:,}행, "
        f"{ambiguous.group_count:,}그룹**입니다.",
        f"- `NO`가 반복되는 행은 **{no_repeat.row_count:,}행, {no_repeat.group_count:,}개 NO값**입니다. "
        "따라서 `NO`는 전체 기간의 고유 거래키로 사용할 수 없습니다.",
        "",
        "## 시나리오별 집계",
        "",
        "`중복행 수`는 중복 그룹에 포함된 모든 행, `초과 행 수`는 각 그룹에서 한 행만 남긴다고 "
        "가정할 때 추가로 존재하는 행 수입니다. 서로 다른 시나리오의 행은 중첩될 수 있습니다.",
        "",
        "| 상황 | 의미 | 중복행 수 | 그룹 수 | 초과 행 수 | 최대 그룹 크기 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.interpretation} | {result.row_count:,} | "
            f"{result.group_count:,} | {result.excess_count:,} | {result.max_group_size:,} |"
        )

    lines.extend(
        [
            "",
            "## 분석 기준",
            "",
            "### 1. 전체 컬럼 완전 동일",
            "",
            "20개 원본 컬럼 값이 모두 같은 행입니다. 발견될 경우 가장 명확한 중복입니다.",
            "",
            "### 2. NO 제외 동일",
            "",
            "기술적 식별자로 보이는 `NO`를 제외한 19개 컬럼이 모두 같은 행입니다. 이번 데이터에서는 "
            "완전 동일행이 없으므로, 이 그룹들은 내용은 같고 `NO`가 다른 강한 중복 후보입니다.",
            "",
            "### 3. 핵심 거래키 동일",
            "",
            "다음 컬럼 조합이 같은 행입니다.",
            "",
            f"`{', '.join(core_keys)}`",
            "",
            "공개 데이터에는 개별 호수가 없으므로 동일 단지·동·층·면적의 여러 세대가 같은 날 같은 가격에 "
            "거래된 정상 사례가 포함될 수 있습니다. 따라서 이 기준만으로 삭제하면 안 됩니다.",
            "",
            "### 4. 핵심키 동일·세부정보 차이",
            "",
            "핵심 거래키는 같지만 `NO` 이외의 나머지 정보 중 하나 이상이 다른 그룹입니다. "
            "추출 CSV에서 그룹별로 어떤 값이 다른지 직접 비교할 수 있습니다.",
            "",
            "| 차이가 발생한 세부 컬럼 | 해당 그룹 수 |",
            "|---|---:|",
        ]
    )
    for column, count in sorted(variations.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {column} | {count:,} |")

    lines.extend(
        [
            "",
            "### 5. NO 반복",
            "",
            "`NO` 하나만 기준으로 반복을 확인한 결과입니다. 같은 `NO`가 서로 다른 거래에 재사용된 경우가 "
            "대부분이므로 거래 중복 증거가 아니라 식별자 품질 점검 결과입니다.",
            "",
            "## 행 확인용 CSV",
            "",
            "각 파일에는 원본 컬럼 앞에 `_중복그룹ID`, `_그룹행수`, `_원본CSV행번호`를 추가했습니다. "
            "같은 그룹이 연속되도록 정렬되어 있습니다.",
            "",
            "| 상황 | 파일 | 행 수 | 파일 크기 | SHA-256 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.name} | `{result.output_path.name}` | {result.row_count:,} | "
            f"{human_bytes(result.output_path.stat().st_size)} | `{result.sha256}` |"
        )

    lines.extend(
        [
            "",
            "## 검증 결과",
            "",
            "- 모든 추출 CSV의 컬럼 구성: **통과**",
            "- 모든 추출 CSV의 행 수 재검산: **통과**",
            "- 추출 파일의 그룹행수 2 이상 여부: **통과**",
            "- 원본 파일 미변경: **통과(SHA-256 작업 전후 동일)**",
            "",
            "## 재현 방법",
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
        raise FileNotFoundError(f"분석할 CSV를 찾을 수 없습니다: {SOURCE_CSV}")

    source_hash_before = sha256_file(SOURCE_CSV)
    frame = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig", dtype="string")
    all_columns = frame.columns.tolist()
    if "NO" not in all_columns:
        raise KeyError("필수 컬럼 `NO`가 없습니다.")

    content_keys = [column for column in all_columns if column != "NO"]
    core_keys = [
        "시군구",
        "번지",
        "단지명",
        "전용면적(㎡)",
        "계약년월",
        "계약일",
        "거래금액(만원)",
        "동",
        "층",
    ]
    missing_core = [column for column in core_keys if column not in all_columns]
    if missing_core:
        raise KeyError(f"핵심 거래키 컬럼이 없습니다: {missing_core}")

    exact_mask = duplicated_mask(frame, all_columns)
    except_no_mask = duplicated_mask(frame, content_keys)
    core_mask = duplicated_mask(frame, core_keys)
    core_different_mask = ambiguous_core_mask(frame, core_mask, core_keys, content_keys)
    no_mask = duplicated_mask(frame, ["NO"])

    scenario_specs = [
        (
            exact_mask,
            "전체 컬럼 완전 동일",
            "20개 원본 컬럼이 모두 동일한 확정 중복",
            all_columns,
            OUTPUT_EXACT,
        ),
        (
            except_no_mask,
            "NO 제외 동일",
            "NO를 제외한 19개 컬럼이 동일한 강한 중복 후보",
            content_keys,
            OUTPUT_EXCEPT_NO,
        ),
        (
            core_mask,
            "핵심 거래키 동일",
            "주소·단지·면적·계약일·금액·동·층이 같은 검토 후보",
            core_keys,
            OUTPUT_CORE,
        ),
        (
            core_different_mask,
            "핵심키 동일·세부정보 차이",
            "핵심키는 같지만 NO 이외 세부정보가 다른 검토 그룹",
            core_keys,
            OUTPUT_CORE_DIFFERENT,
        ),
        (
            no_mask,
            "NO 반복",
            "NO 값의 재사용 여부이며 거래 중복 판정 기준은 아님",
            ["NO"],
            OUTPUT_NO,
        ),
    ]

    results = [
        extract_scenario(
            frame,
            mask,
            name=name,
            interpretation=interpretation,
            keys=keys,
            output_path=output_path,
        )
        for mask, name, interpretation, keys, output_path in scenario_specs
    ]

    variations = varying_columns_by_core_group(
        frame, core_different_mask, core_keys, content_keys
    )
    for result in results:
        validate_extract(result, all_columns)

    source_hash_after = sha256_file(SOURCE_CSV)
    if source_hash_before != source_hash_after:
        raise RuntimeError("분석 도중 원본 CSV가 변경되었습니다.")

    write_report(
        frame=frame,
        source_hash=source_hash_before,
        results=results,
        core_keys=core_keys,
        variations=variations,
    )

    print(f"분석 완료: {len(frame):,}행 × {len(all_columns)}열")
    for result in results:
        print(
            f"{result.name}: {result.row_count:,}행, "
            f"{result.group_count:,}그룹, 초과 {result.excess_count:,}행"
        )
    print(f"보고서: {REPORT_MD}")


if __name__ == "__main__":
    main()
