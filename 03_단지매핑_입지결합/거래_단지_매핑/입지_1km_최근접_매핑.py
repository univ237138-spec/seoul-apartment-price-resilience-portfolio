"""Keep one nearest facility per type and verify against every source facility."""

import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('facility_mapping', HERE / '입지_1km_매핑.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ROOT = HERE.parents[1]
COMPLEX = ROOT / '01_원천데이터/아파트_단지_DATA/아파트_단지_DATA_CLEAN_final.csv'
PREVIOUS = base.OUT_CSV
OUT_DIR = ROOT / '거래_단지_DATA/입지_1km_최근접_매핑_결과'
OUT_CSV = OUT_DIR / '아파트_거래_DATA_상위5000행_입지시설_1km_최근접.csv'
OUT_MD = OUT_DIR / '아파트_입지시설_1km_최근접_매핑_결과.md'
TYPES = ('학교', '지하철', '공원', '대규모점포')


def main():
    sources = [base.APARTMENTS, COMPLEX, base.FACILITIES, PREVIOUS, base.OUT_MD]
    hashes = {path: base.hash_file(path) for path in sources}
    apt_fields, apartments = base.read_csv(base.APARTMENTS)
    facility_fields, facilities = base.read_csv(base.FACILITIES)
    complex_fields, complexes = base.read_csv(COMPLEX)
    complex_by_code = {row['KAPT단지코드']: row for row in complexes}
    assert len(complex_by_code) == len(complexes)
    assert all('단지_' + field in apt_fields for field in complex_fields)
    complex_matched = 0
    for row in apartments:
        match = complex_by_code.get(row['단지_KAPT단지코드'])
        if match is not None:
            complex_matched += 1
            assert all(row['단지_' + field] == match[field] for field in complex_fields)

    suffixes = facility_fields + ['거리_m', '입지원본행번호']
    output_fields = apt_fields + ['입지매핑상태'] + [f'{kind}_{suffix}' for kind in TYPES for suffix in suffixes]
    assert len(set(output_fields)) == len(output_fields)
    facility_points = [(i, base.coordinates(row)[0]) for i, row in enumerate(facilities)]
    facility_points = [(i, point) for i, point in facility_points if point is not None]
    # Recompute minimum over ALL facilities without relying on previous ranking.
    points = [base.coordinates(row) for row in apartments]
    cache = {}
    checked_pairs = 0
    for point, status in points:
        if point is None or point in cache:
            continue
        nearest = {kind: None for kind in TYPES}
        for index, target in facility_points:
            distance = base.haversine_m(point, target)
            checked_pairs += 1
            kind = facilities[index]['시설유형']
            if distance <= base.LIMIT_M:
                candidate = (distance, index)
                if nearest[kind] is None or candidate < nearest[kind]:
                    nearest[kind] = candidate
        cache[point] = nearest
    print(f'Nearest facilities verified over {checked_pairs:,} coordinate pairs.', flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    counters = Counter()
    statuses = Counter()
    examples = []
    missing = []
    all_four = any_facility = 0
    with PREVIOUS.open(encoding='utf-8-sig', newline='') as old_handle, OUT_CSV.open('w', encoding='utf-8-sig', newline='') as out_handle:
        previous_reader = csv.DictReader(old_handle)
        previous_column_count = len(previous_reader.fieldnames)
        writer = csv.DictWriter(out_handle, fieldnames=output_fields)
        writer.writeheader()
        for index, (row, (point, status), previous) in enumerate(zip(apartments, points, previous_reader, strict=True), 2):
            assert all(previous[field] == row[field] for field in apt_fields)
            assert previous['입지매핑상태'] == status
            result = dict(row, 입지매핑상태=status)
            statuses[status] += 1
            selected_count = 0
            for kind in TYPES:
                candidate = cache[point][kind] if point is not None else None
                values = {suffix: '' for suffix in suffixes}
                if candidate is not None:
                    distance, facility_index = candidate
                    values.update(facilities[facility_index])
                    values['거리_m'] = f'{distance:.3f}'
                    values['입지원본행번호'] = str(facility_index + 2)
                    counters[kind] += 1
                    selected_count += 1
                for suffix, value in values.items():
                    assert previous[f'{kind}_01_{suffix}'] == value
                    result[f'{kind}_{suffix}'] = value
            writer.writerow(result)
            all_four += selected_count == 4
            any_facility += selected_count > 0
            if index == 2:
                examples.append(result)
            if point is None:
                missing.append((index, row['NO'], row['단지명'], status))

    # Read the final file back, comparing every input and output field.
    checked_rows = 0
    with OUT_CSV.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == output_fields
        for original, (point, status), actual in zip(apartments, points, reader, strict=True):
            assert None not in actual and all(value is not None for value in actual.values())
            assert all(actual[field] == original[field] for field in apt_fields)
            assert actual['입지매핑상태'] == status
            for kind in TYPES:
                candidate = cache[point][kind] if point is not None else None
                if candidate is None:
                    assert all(actual[f'{kind}_{suffix}'] == '' for suffix in suffixes)
                else:
                    distance, facility_index = candidate
                    assert all(actual[f'{kind}_{field}'] == facilities[facility_index][field] for field in facility_fields)
                    assert actual[f'{kind}_거리_m'] == f'{distance:.3f}'
                    assert actual[f'{kind}_입지원본행번호'] == str(facility_index + 2)
                    assert distance <= base.LIMIT_M
            checked_rows += 1
    assert checked_rows == len(apartments)
    assert all(base.hash_file(path) == digest for path, digest in hashes.items())

    report = [
        '# 아파트 입지시설 1km 이내 유형별 최근접 1개 매핑 결과', '',
        f'- 생성 시각: {datetime.now().astimezone().isoformat(timespec="seconds")}',
        f'- 결과 CSV: [{OUT_CSV.name}]({OUT_CSV.name})',
        f'- 결과: **{checked_rows:,}행 × {len(output_fields):,}열**. 기존 전체 시설 매핑 {previous_column_count:,}열에서 {previous_column_count - len(output_fields):,}열 축소.',
        f'- 원본 거래 {len(apt_fields)}열 + 입지매핑상태 1열 + 유형 4개 × 시설 정보 8열.',
        '- 원본 거래 순서와 모든 문자열 값(선행 0 포함)을 보존했습니다. UTF-8 BOM CSV로 저장했습니다.', '',
        '## 1. 사용 데이터와 단지 정보 처리', '',
        f'- 기준 거래: `{base.APARTMENTS.relative_to(ROOT).as_posix()}`',
        f'- 단지 참조: `{COMPLEX.relative_to(ROOT).as_posix()}`',
        f'- 시설 원본: `{base.FACILITIES.name}`',
        f'- 기존 매핑 기준: `{base.OUT_MD.relative_to(ROOT).as_posix()}`',
        f'- 기존 전체 매핑 결과: `{PREVIOUS.relative_to(ROOT).as_posix()}`',
        f'- 단지 참조 파일은 {len(complexes):,}행·{len(complex_fields)}열이며 위도·경도 컬럼이 없습니다. 거래 데이터에는 이 단지 정보가 이미 `단지_` 접두사로 포함되어 있습니다.',
        f'- `단지_KAPT단지코드`로 참조되는 거래 {complex_matched:,}행의 단지 정보 {len(complex_fields)}개 컬럼이 단지 파일과 모두 일치함을 확인했습니다. 나머지 {len(apartments) - complex_matched:,}행도 원본 단지 매칭 상태와 값을 그대로 유지했습니다.',
        '- 시설 매핑은 거래 데이터의 `위도`, `경도`를 사용합니다. 단지 파일의 주소로 새 좌표를 조회하거나 단지 정보를 다시 매칭하지 않았습니다.', '',
        '## 2. 최근접 선택 기준', '',
        '- 학교·지하철·공원·대규모점포 각각에 대해 1,000m 이하 시설 중 거리가 최소인 원본 시설 행 1개만 선택합니다. 한 거래 행에는 최대 4개 시설이 연결됩니다.',
        '- 기존 MD와 동일하게 Haversine 공식과 평균 지구 반지름 6,371,008.8m를 사용했습니다. 두 원본의 십진도 좌표가 동일한 지리 좌표계라고 가정합니다.',
        '- 거리는 원본 대표 좌표 사이의 구면상 최단거리입니다. 도보·차량 이동거리, 출입구·공원 경계까지의 거리가 아닙니다.',
        '- 1km 포함 여부와 최근접 순위는 반올림 전 거리로 결정합니다. 거리가 정확히 같으면 시설 원본 행 번호가 앞선 1개를 선택합니다.',
        '- `거리_m`은 미터 단위, 소수점 셋째 자리까지 기록합니다.',
        '- 학교는 초등학교·중학교·고등학교를 통틀어 가장 가까운 1개입니다. 학교급별로 각각 선택하지 않습니다.',
        '- 지하철 복수 노선은 선택된 원본 행의 `세부유형` 문자열을 그대로 보존합니다.',
        '- 해당 유형의 1km 내 시설이 없으면 해당 유형의 8개 컬럼은 모두 빈값입니다. 좌표가 없는 행은 `입지매핑상태=좌표없음`으로 두고 시설 정보를 모두 비웁니다.',
        '- 시설 원본에서 주소·세부유형이 빈값이면 결과에서도 빈값을 유지합니다. 시설 운영 시점과 거래 시점은 매핑 조건에 포함되지 않습니다.', '',
        '```text',
        'a = sin²((φ₂−φ₁)/2) + cos(φ₁) × cos(φ₂) × sin²((λ₂−λ₁)/2)',
        '거리_m = 2 × 6371008.8 × asin(√a)  (φ, λ는 라디안)',
        '유형별 선택 = 거리_m ≤ 1000인 시설 중 (거리_m, 원본 행 번호)가 최소인 1개',
        '```', '',
        '## 3. 컬럼 변경', '',
        '- 시설 개수 컬럼 4개와 각 유형의 두 번째 이후 시설 컬럼을 결과에서 제거했습니다.',
        '- 첫 번째 시설 컬럼의 `_01_`을 없앴습니다. 예: `학교_01_시설명` → `학교_시설명`, `학교_01_거리_m` → `학교_거리_m`.',
        '- 컬럼 순서: 원본 거래·단지 48열 → 입지매핑상태 → 학교 8열 → 지하철 8열 → 공원 8열 → 대규모점포 8열.', '',
        '| 유형별 컬럼 패턴 | 내용 |', '|---|---|',
        '| {유형}_시설유형 | 원본 시설유형 |',
        '| {유형}_시설명 | 1km 내 가장 가까운 시설 이름 |',
        '| {유형}_세부유형 | 학교급 또는 지하철 노선 등 원본값 |',
        '| {유형}_주소 | 시설 주소 원본값 |',
        '| {유형}_위도 | 시설 위도 원본값 |',
        '| {유형}_경도 | 시설 경도 원본값 |',
        '| {유형}_거리_m | 해당 거래 아파트에서 선택한 시설까지 거리(m) |',
        '| {유형}_입지원본행번호 | 헤더를 1행으로 보는 시설 CSV 레코드 번호(데이터 첫 행=2) |', '',
        '## 4. 처리 결과', '',
        '| 항목 | 행 수 |', '|---|---:|',
        f'| 전체 입력/출력 | {checked_rows:,} |',
        f'| 좌표 정상 | {statuses["정상"]:,} |',
        f'| 좌표 없음 | {statuses["좌표없음"]:,} |',
        f'| 좌표 오류 | {statuses["좌표오류"]:,} |',
        f'| 한 유형 이상 매핑 | {any_facility:,} |',
        f'| 네 유형 모두 매핑 | {all_four:,} |',
        f'| 좌표 정상이나 모든 유형의 1km 내 시설 없음 | {statuses["정상"] - any_facility:,} |', '',
        '| 시설유형 | 최근접 시설이 매핑된 거래 행 | 좌표 정상이나 1km 내 해당 유형 없음 |', '|---|---:|---:|',
    ]
    for kind in TYPES:
        report.append(f'| {kind} | {counters[kind]:,} | {statuses["정상"] - counters[kind]:,} |')
    report += ['', '집계는 거래 행 기준입니다. 같은 아파트의 여러 거래에 같은 시설이 연결되면 각 거래를 별도로 셉니다.', '',
               '## 5. 결과 예시', '']
    for example in examples:
        report += [f'아파트: **{example["단지명"]}**, NO `{example["NO"]}`, 좌표 `{example["위도"]}, {example["경도"]}`', '',
                   '| 유형 | 선택된 시설명 | 세부유형 | 거리(m) |', '|---|---|---|---:|']
        for kind in TYPES:
            report.append('| ' + ' | '.join(base.md_cell(value) for value in [kind, example[f'{kind}_시설명'], example[f'{kind}_세부유형'], example[f'{kind}_거리_m']]) + ' |')
    report += ['', '## 6. 좌표 미보유·오류 행', '',
               '| 거래 CSV 원본 행 | NO | 단지명 | 상태 |', '|---:|---|---|---|']
    for values in missing:
        report.append('| ' + ' | '.join(base.md_cell(value) for value in values) + ' |')
    report += ['', '## 7. 검증 결과', '',
               f'- 유효 고유 아파트 좌표 {len(cache):,}개와 유효 시설 {len(facility_points):,}개의 모든 조합 {checked_pairs:,}쌍을 재계산하고 유형별 최소 거리를 선택했습니다.',
               '- 재계산 결과와 기존 전체 매핑의 각 유형 `_01_` 컬럼 8개가 모든 거래 행에서 일치했습니다.',
               f'- 최종 CSV를 다시 읽어 {checked_rows:,}행 전체의 원본 값·행 순서·매핑 상태·시설 원본값·거리·출처 행 번호·빈값·열 수를 검증했습니다: 통과.',
               '- 각 유형은 0개 또는 1개만 매핑되며, 선택된 시설은 반올림 전 거리 1,000m 이하입니다.',
               '- 참조한 파일의 처리 전후 SHA-256 일치를 확인했습니다. 원본과 이전 결과는 변경하지 않았습니다.', '',
               '| 파일 | SHA-256 |', '|---|---|']
    for path, digest in hashes.items():
        report.append(f'| {path.name} | `{digest}` |')
    report.append(f'| {OUT_CSV.name} | `{base.hash_file(OUT_CSV)}` |')
    OUT_MD.write_text('\n'.join(report) + '\n', encoding='utf-8-sig')
    print(json.dumps({'csv': str(OUT_CSV), 'md': str(OUT_MD), 'rows': checked_rows,
                      'columns': len(output_fields), 'removed_columns': previous_column_count - len(output_fields),
                      'matched_rows_by_type': counters, 'status': statuses, 'all_four': all_four}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
