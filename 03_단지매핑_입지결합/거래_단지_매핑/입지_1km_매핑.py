"""Map all nearby facility records to the original apartment rows, without dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APARTMENTS = ROOT / '거래_단지_DATA/아파트_거래_DATA_단지DATA_매핑_상위5000행_위도경도.csv'
FACILITIES = ROOT / '01_원천데이터/아파트_입지_DATA.csv'
OUT_DIR = ROOT / '거래_단지_DATA/입지_1km_매핑_결과'
OUT_CSV = OUT_DIR / '아파트_거래_DATA_상위5000행_입지시설_1km_매핑.csv'
OUT_MD = OUT_DIR / '아파트_입지시설_1km_매핑_결과.md'
TYPES = ('학교', '지하철', '대규모점포', '공원')
EARTH_RADIUS_M = 6_371_008.8
LIMIT_M = 1_000.0


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        assert fields and len(fields) == len(set(fields)), 'Duplicate or missing headers'
        rows = list(reader)
    assert all(None not in r and all(v is not None for v in r.values()) for r in rows)
    return fields, rows


def coordinates(row):
    if not row['위도'].strip() or not row['경도'].strip():
        return None, '좌표없음'
    try:
        latitude, longitude = float(row['위도']), float(row['경도'])
    except ValueError:
        return None, '좌표오류'
    if not (math.isfinite(latitude) and math.isfinite(longitude)
            and -90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None, '좌표오류'
    return (latitude, longitude), '정상'


def haversine_m(first, second):
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def unit_vector(point):
    lat, lon = map(math.radians, point)
    return math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def main():
    apt_fields, apartments = read_csv(APARTMENTS)
    facility_fields, facilities = read_csv(FACILITIES)
    assert facility_fields == ['시설유형', '시설명', '세부유형', '주소', '위도', '경도']
    assert set(r['시설유형'] for r in facilities) <= set(TYPES)
    initial_hashes = {p: hash_file(p) for p in (APARTMENTS, FACILITIES)}

    facility_points = [coordinates(r) for r in facilities]
    valid_facilities = [(i, p) for i, (p, _) in enumerate(facility_points) if p is not None]
    apartment_points = [coordinates(r) for r in apartments]
    unique_points = list(dict.fromkeys(p for p, _ in apartment_points if p is not None))
    cache = {}
    for point in unique_points:
        matches = {t: [] for t in TYPES}
        for index, target in valid_facilities:
            distance = haversine_m(point, target)
            if distance <= LIMIT_M:
                matches[facilities[index]['시설유형']].append((distance, index))
        for values in matches.values():
            values.sort(key=lambda item: (item[0], item[1]))
        cache[point] = matches
    print(f'Matched {len(unique_points):,} unique apartment coordinates.', flush=True)

    maxima = {t: max((len(m[t]) for m in cache.values()), default=0) for t in TYPES}
    suffixes = facility_fields + ['거리_m', '입지원본행번호']
    output_fields = apt_fields + ['입지매핑상태']
    for kind in TYPES:
        output_fields.append(f'{kind}_1km이내_개수')
        for rank in range(1, maxima[kind] + 1):
            output_fields.extend(f'{kind}_{rank:02d}_{suffix}' for suffix in suffixes)
    assert len(output_fields) == len(set(output_fields))

    OUT_DIR.mkdir(exist_ok=True)
    with OUT_CSV.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(output_fields)
        for row, (point, status) in zip(apartments, apartment_points):
            groups = cache.get(point)
            result = [row[k] for k in apt_fields] + [status]
            for kind in TYPES:
                matches = groups[kind] if groups else []
                result.append(len(matches) if groups is not None else '')
                for distance, index in matches:
                    result.extend(facilities[index][k] for k in facility_fields)
                    result.extend([f'{distance:.3f}', index + 2])
                result.extend([''] * ((maxima[kind] - len(matches)) * len(suffixes)))
            assert len(result) == len(output_fields)
            writer.writerow(result)

    # Independently verify the complete search using the 3D chord length,
    # including excluded pairs, to catch missed facilities and threshold errors.
    vectors = [(index, unit_vector(point)) for index, point in valid_facilities]
    checked_pairs = 0
    max_difference = 0.0
    for point, groups in cache.items():
        x, y, z = unit_vector(point)
        independently_found = set()
        expected = {index: distance for matches in groups.values() for distance, index in matches}
        for index, (a, b, c) in vectors:
            chord = math.sqrt((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2)
            distance = 2 * EARTH_RADIUS_M * math.asin(min(1.0, chord / 2))
            checked_pairs += 1
            if distance <= LIMIT_M:
                independently_found.add(index)
                difference = abs(distance - expected[index])
                max_difference = max(max_difference, difference)
                assert difference < 0.000001
        assert independently_found == set(expected), 'Distance membership mismatch'

    # Re-read the delivered CSV and verify every original field and added cell.
    with OUT_CSV.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == output_fields
        verified_rows = 0
        for original, (point, status), result in zip(apartments, apartment_points, reader, strict=True):
            assert None not in result and all(v is not None for v in result.values())
            assert all(result[k] == original[k] for k in apt_fields)
            assert result['입지매핑상태'] == status
            groups = cache.get(point)
            for kind in TYPES:
                matches = groups[kind] if groups else []
                assert result[f'{kind}_1km이내_개수'] == (str(len(matches)) if groups is not None else '')
                for rank in range(1, maxima[kind] + 1):
                    prefix = f'{kind}_{rank:02d}_'
                    if rank <= len(matches):
                        distance, index = matches[rank - 1]
                        assert all(result[prefix + k] == facilities[index][k] for k in facility_fields)
                        assert result[prefix + '거리_m'] == f'{distance:.3f}'
                        assert result[prefix + '입지원본행번호'] == str(index + 2)
                    else:
                        assert all(result[prefix + k] == '' for k in suffixes)
            verified_rows += 1
    assert verified_rows == len(apartments)
    assert all(hash_file(p) == expected for p, expected in initial_hashes.items())

    valid_rows = sum(p is not None for p, _ in apartment_points)
    status_counts = Counter(status for _, status in apartment_points)
    source_counts = Counter(r['시설유형'] for r in facilities)
    statistics = {}
    for kind in TYPES:
        counts = [len(cache[p][kind]) for p, _ in apartment_points if p is not None]
        ids = {index for groups in cache.values() for _, index in groups[kind]}
        statistics[kind] = {
            'source_facilities': source_counts[kind],
            'matched_apartment_rows': sum(count > 0 for count in counts),
            'total_matches': sum(counts),
            'distinct_source_facilities': len(ids),
            'max_per_row': maxima[kind],
            'mean_per_valid_row': sum(counts) / valid_rows if valid_rows else 0,
        }
    all_four = sum(all(cache[p][t] for t in TYPES) for p, _ in apartment_points if p is not None)
    any_match = sum(any(cache[p][t] for t in TYPES) for p, _ in apartment_points if p is not None)
    exact_duplicates = len(facilities) - len({tuple(r[k] for k in facility_fields) for r in facilities})

    report = [
        '# 아파트 입지시설 1km 매핑 결과', '',
        f'- 생성 시각: {datetime.now().astimezone().isoformat(timespec="seconds")}',
        f'- 기준 아파트: `{APARTMENTS.relative_to(ROOT).as_posix()}`',
        f'- 입지 원본: `{FACILITIES.name}`',
        f'- 결과 CSV: [{OUT_CSV.name}]({OUT_CSV.name})',
        f'- 결과 크기: **{len(apartments):,}행 × {len(output_fields):,}열** (헤더 제외). 원본 {len(apt_fields)}열과 행 순서·문자열 값을 보존하고 {len(output_fields) - len(apt_fields):,}열 추가.',
        '- CSV 인코딩: UTF-8 BOM, 쉼표 구분. 쉼표·줄바꿈을 포함한 원본 값은 CSV 인용 규칙으로 보존.', '',
        '## 1. 거리 계산 및 매핑 기준', '',
        '- 두 파일의 `위도`, `경도`를 십진도 좌표로 해석했습니다. 원본에 좌표계 메타데이터가 없으므로 동일한 지리 좌표계라고 가정했습니다.',
        f'- Haversine 공식, 평균 지구 반지름 R = {EARTH_RADIUS_M:,.1f}m를 사용한 구면상 최단거리입니다. 도보·차량 이동거리나 시설 경계·출입구까지의 거리가 아닌 원본 대표 좌표 간 거리입니다.',
        '- 반올림 전 계산값이 **1,000m 이하(경계 포함)**인 학교·지하철·대규모점포·공원 원본 행을 모두 가져왔습니다. 최근접 1개 제한은 없습니다.',
        '- 저장한 거리 단위는 m이며 소수점 셋째 자리까지 반올림합니다. 표시 정밀도가 원본 좌표의 실제 위치 정확도를 뜻하지는 않습니다.',
        '- 유형별 거리 오름차순으로 번호를 부여합니다. 거리가 정확히 같으면 입지 원본 행 순서를 따릅니다. 같은 번호도 아파트 행에 따라 다른 시설일 수 있습니다.',
        '- 시설명이나 좌표가 같아도 원본의 서로 다른 행은 각각 보존합니다. 지하철의 복수 노선 정보는 `세부유형` 원본 문자열 그대로 유지합니다.',
        f'- 입지 원본 완전 중복행: {exact_duplicates:,}개. 유효 좌표 시설: {len(valid_facilities):,}/{len(facilities):,}개.',
        '- 같은 좌표를 가진 거래 행은 계산 결과를 재사용하되 모든 거래 행을 유지합니다. 거래 시점과 시설의 개설·폐업 시점은 매핑 조건에 포함되지 않습니다.', '',
        '```text',
        'φ = 위도를 라디안으로 변환, λ = 경도를 라디안으로 변환',
        'a = sin²((φ₂−φ₁)/2) + cos(φ₁) × cos(φ₂) × sin²((λ₂−λ₁)/2)',
        '거리_m = 2 × R × asin(√a)',
        '매핑 조건: 거리_m ≤ 1000 (반올림 전)',
        '```', '',
        '## 2. 처리 요약', '',
        '| 항목 | 결과 |', '|---|---:|',
        f'| 입력/출력 아파트 행 수 | {len(apartments):,} |',
        f'| 좌표 정상 행 | {valid_rows:,} |',
        f'| 좌표 없음 행 | {status_counts["좌표없음"]:,} |',
        f'| 좌표 오류 행 | {status_counts["좌표오류"]:,} |',
        f'| 유효 고유 아파트 좌표 | {len(unique_points):,} |',
        f'| 한 유형 이상 시설이 매핑된 행 | {any_match:,} |',
        f'| 네 유형이 모두 매핑된 행 | {all_four:,} |',
        f'| 좌표는 있으나 네 유형 모두 1km 내 시설이 없는 행 | {valid_rows - any_match:,} |', '',
        '| 시설유형 | 원본 시설 수 | 매핑된 아파트 행 | 전체 연결 건수 | 연결된 서로 다른 원본 시설 | 행당 최대 | 좌표 정상 행당 평균 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for kind, stats in statistics.items():
        report.append(f'| {kind} | {stats["source_facilities"]:,} | {stats["matched_apartment_rows"]:,} | {stats["total_matches"]:,} | {stats["distinct_source_facilities"]:,} | {stats["max_per_row"]:,} | {stats["mean_per_valid_row"]:.3f} |')
    report += [
        '', '전체 연결 건수는 아파트 거래 행 × 시설 원본 행 기준입니다. 같은 아파트의 여러 거래에 동일한 시설이 연결되면 각각 계산합니다. 원본 시설 수는 현재 제공된 입지 데이터의 범위를 뜻합니다.', '',
        '## 3. 추가 컬럼 구성', '',
        '원본 48열 → `입지매핑상태` → 학교 그룹 → 지하철 그룹 → 대규모점포 그룹 → 공원 그룹 순서입니다. 각 유형 그룹은 개수 컬럼 뒤에 시설별 컬럼 묶음이 이어집니다.', '',
        '| 컬럼 또는 접미사 | 의미 |', '|---|---|',
        '| 입지매핑상태 | 정상 / 좌표없음 / 좌표오류 |',
        '| {유형}_1km이내_개수 | 해당 거래 행에 연결된 해당 유형의 입지 원본 행 개수 |',
        '| {유형}_{번호}_시설유형 | 시설유형 원본값 |',
        '| {유형}_{번호}_시설명 | 시설명 원본값 |',
        '| {유형}_{번호}_세부유형 | 학교급 또는 지하철 노선 등 원본값 |',
        '| {유형}_{번호}_주소 | 주소 원본값 |',
        '| {유형}_{번호}_위도 | 시설 위도 원본값 |',
        '| {유형}_{번호}_경도 | 시설 경도 원본값 |',
        '| {유형}_{번호}_거리_m | 해당 아파트 좌표와 해당 시설 좌표 간 거리(m) |',
        '| {유형}_{번호}_입지원본행번호 | 입지 CSV의 헤더를 1행으로 보는 레코드 행 번호. 데이터 첫 행은 2 |', '',
        '| 시설 그룹 | 번호 범위 | 추가 열 수(개수 포함) |', '|---|---|---:|',
    ]
    for kind in TYPES:
        report.append(f'| {kind} | 01–{maxima[kind]:02d} | {1 + maxima[kind] * len(suffixes):,} |')
    report += [
        '', '- 예: `학교_01_시설명`, `학교_01_세부유형`, …, `학교_01_거리_m`은 같은 학교에 대한 정보입니다.',
        '- 1km 이내 시설이 없으면 해당 유형의 개수는 0, 시설별 정보는 빈값입니다. 거리 0은 실제 좌표가 같은 경우에만 기록됩니다.',
        '- 좌표가 없거나 잘못된 행은 계산하지 않았으므로 시설 개수와 모든 시설별 정보를 빈값으로 두었습니다.',
        '- 각 유형은 전체 행에서 관측된 최대 시설 수만큼 컬럼을 만들며, 개별 행의 남는 번호 묶음은 빈값입니다.',
        '- 입지 원본 결측값을 추정하거나 채우지 않았습니다. 원본의 주소 결측은 '
        f'{sum(not r["주소"] for r in facilities):,}행, 세부유형 결측은 {sum(not r["세부유형"] for r in facilities):,}행입니다.',
        '- CSV는 문자열 타입 정보를 저장하지 않습니다. Excel 등에서 열 때 본번·부번 등 선행 0을 보존하려면 해당 열을 텍스트로 가져오십시오.', '',
        '## 4. 좌표 미보유·오류 행', '',
        '| 아파트 CSV 원본 행 번호 | NO | 단지명 | 상태 |', '|---:|---|---|---|',
    ]
    for number, (row, (point, status)) in enumerate(zip(apartments, apartment_points), 2):
        if point is None:
            report.append(f'| {number} | {md_cell(row["NO"])} | {md_cell(row["단지명"])} | {status} |')
    report += ['', '## 5. 매핑 예시', '']
    example_indices = list(dict.fromkeys([0] + [i for i, (p, _) in enumerate(apartment_points) if p is not None and all(cache[p][t] for t in TYPES)][:1]))
    for i in example_indices:
        row = apartments[i]
        point, status = apartment_points[i]
        report += [f'### 아파트 원본 {i + 2}행: {md_cell(row["단지명"])} (NO {row["NO"]})', '',
                   f'- 아파트 좌표: {row["위도"]}, {row["경도"]}; 상태: {status}', '',
                   '| 유형 | 1km 내 전체 개수 | 순번 | 시설명 | 세부유형 | 거리(m) | 입지 원본 행 |',
                   '|---|---:|---:|---|---|---:|---:|']
        if point is not None:
            for kind in TYPES:
                matches = cache[point][kind]
                if not matches:
                    report.append(f'| {kind} | 0 | — | — | — | — | — |')
                for rank, (distance, index) in enumerate(matches[:3], 1):
                    facility = facilities[index]
                    report.append(f'| {kind} | {len(matches)} | {rank:02d} | {md_cell(facility["시설명"])} | {md_cell(facility["세부유형"])} | {distance:.3f} | {index + 2} |')
        report += ['', '보고서 예시는 유형별 가까운 3개까지만 표시하며, 결과 CSV에는 조건을 만족하는 모든 시설과 원본 6개 컬럼이 들어 있습니다.', '']
    report += [
        '## 6. 검증 결과', '',
        f'- 저장된 CSV를 다시 읽어 {verified_rows:,}행 전체의 원본 {len(apt_fields)}개 컬럼 값, 행 순서, 추가 컬럼 값, 빈 슬롯, 헤더와 행 너비를 검증했습니다: 통과.',
        f'- 유효 고유 아파트 좌표 × 유효 시설 전체 {checked_pairs:,}쌍을 독립적인 3차원 단위구 현 길이 방식으로 재계산하여 포함/제외 시설 집합이 정확히 같은지 검증했습니다: 통과.',
        f'- 매핑된 쌍의 두 계산 방식 간 최대 거리 차이: {max_difference:.12f}m. 허용 검증 오차 0.000001m 미만: 통과.',
        '- 모든 매핑은 반올림 전 1,000m 이하이고 시설 유형별 거리순 정렬을 유지합니다.',
        '- 원본 두 파일의 처리 전후 SHA-256이 동일합니다: 원본 변경 없음.', '',
        '| 파일 | SHA-256 |', '|---|---|',
    ]
    for path, digest in initial_hashes.items():
        report.append(f'| {path.name} | `{digest}` |')
    report.append(f'| {OUT_CSV.name} | `{hash_file(OUT_CSV)}` |')
    OUT_MD.write_text('\n'.join(report) + '\n', encoding='utf-8-sig')
    print(json.dumps({
        'csv': str(OUT_CSV), 'md': str(OUT_MD), 'rows': len(apartments),
        'columns': len(output_fields), 'csv_bytes': OUT_CSV.stat().st_size,
        'status': status_counts, 'all_four_types': all_four, 'any_match': any_match,
        'statistics': statistics, 'verified_pairs': checked_pairs,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
