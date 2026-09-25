"""EDA의 지표 계산이 대시보드 화면과 같은 값을 내는지 확인한다.

compute_metrics는 build_snapshot을 벡터화해 다시 쓴 것이므로, 무작위 조합에 대해
실제 build_snapshot 결과와 대조한다. 여기서 어긋나면 이후 EDA 해석 전체가 무의미하다.
"""
import sys

import numpy as np
import pandas as pd

import eda_common as ec

sys.path.insert(0, str(ec.ROOT / "대시보드"))
from housing.metrics import build_snapshot
from housing.models import Dataset

FIELDS = ["price", "quarter_count", "mdd", "peak", "trough", "gap", "recovery", "recovery_quarters"]


def main():
    m = ec.meta()
    reference = pd.Period(m["reference"], freq="Q")
    quarterly, complexes, facilities = ec.load("quarterly"), ec.load("complexes"), ec.load("facilities")
    data = Dataset("real", quarterly, complexes, facilities,
                   pd.Timestamp(m["start"]), pd.Timestamp(m["end"]), reference, m["partial"], m["diagnostics"])

    table = ec.compute_metrics(quarterly, reference).set_index(["complex_id", "area_group"])
    print(f"지표 산출 조합: {len(table):,}")

    rng = np.random.default_rng(0)
    combos = quarterly[["complex_id", "area_group"]].drop_duplicates()
    sample = combos.iloc[rng.choice(len(combos), 400, replace=False)]

    mismatches, in_table, out_table = [], 0, 0
    for cid, area in sample.itertuples(index=False, name=None):
        snap = build_snapshot(data, cid, area)
        if (cid, area) in table.index:
            # 지표 산출 조합: 표시되는 8개 항목이 모두 같은 값이어야 한다.
            in_table += 1
            row = table.loc[(cid, area)]
            for field in FIELDS:
                expected = snap.metrics[field].value
                actual = row.get(field, None)
                actual = None if actual is None or pd.isna(actual) else float(actual)
                if (expected is None) != (actual is None):
                    mismatches.append((cid, area, field, expected, actual))
                elif expected is not None and not np.isclose(float(expected), actual, rtol=1e-9, atol=1e-9):
                    mismatches.append((cid, area, field, expected, actual))
        else:
            # 표에 없는 조합: 대시보드에서도 낙폭·회복 지표가 비어 있어야 한다.
            out_table += 1
            for field in ["mdd", "gap", "recovery"]:
                if snap.metrics[field].value is not None:
                    mismatches.append((cid, area, field, snap.metrics[field].value, "표에 없음"))

    if mismatches:
        print(f"\n불일치 {len(mismatches)}건 (앞 20건)")
        for row in mismatches[:20]:
            print("  ", row)
        raise SystemExit(1)
    print(f"검증 통과: 무작위 {len(sample)}개 조합")
    print(f"  지표 산출 조합 {in_table}개 × {len(FIELDS)}개 항목이 build_snapshot과 일치")
    print(f"  미산출 조합 {out_table}개는 대시보드에서도 낙폭·회복 지표가 비어 있음")


if __name__ == "__main__":
    main()
