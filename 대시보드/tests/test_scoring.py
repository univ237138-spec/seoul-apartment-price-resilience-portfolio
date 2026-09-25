"""Grading is peer-relative, so these tests fix the ranking rules rather than any cutoff."""
from dataclasses import replace
import unittest

import pandas as pd

from test_dashboard import fixture
from housing.metrics import build_snapshot
from housing.models import AREA_LABELS, Dataset, duration
from housing.scoring import DEFENSE_WEIGHT, RECOVERY_WEIGHT, build_scores, grade_of, rank_percentile
from housing.ui import recovery_span, score_details

AREA = AREA_LABELS[2]


def population(shapes):
    """One quarterly table holding several complexes over the same five quarters."""
    frames = []
    for name, prices in shapes.items():
        frames.append(pd.DataFrame({
            "complex_id": name, "area_group": AREA,
            "quarter": pd.period_range("2020Q1", periods=len(prices), freq="Q"),
            "price": prices, "count": 3, "complete": True}))
    return pd.concat(frames, ignore_index=True)


class RankingTests(unittest.TestCase):
    def test_percentile_beats_peers_and_splits_ties(self):
        peers = pd.Series([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(rank_percentile(peers, 5.0), 0.0)
        self.assertEqual(rank_percentile(peers, 50.0), 100.0)
        self.assertEqual(rank_percentile(peers, 20.0), 37.5)  # 1 beaten + half of one tie
        self.assertEqual(rank_percentile(pd.Series([], dtype=float), 1.0), 100.0)

    def test_grade_bands_name_a_rank(self):
        self.assertEqual(grade_of(100), "A+")
        self.assertEqual(grade_of(90), "A+")
        self.assertEqual(grade_of(89.9), "A")
        self.assertEqual(grade_of(0), "C")

    def test_weights_are_twenty_eighty_and_recovery_dominates(self):
        # 얕게 빠졌지만 회복이 나쁜 단지 vs 깊게 빠졌지만 전고점을 넘긴 단지.
        quarterly = population({
            "얕은하락": [100, 100, 95, 95, 96],    # mdd -5%, gap -4%
            "깊은하락": [100, 100, 70, 90, 130],   # mdd -30%, gap +30%
            "중간": [100, 100, 85, 90, 100],
        })
        scores = build_scores(quarterly, pd.Period("2021Q1", freq="Q"), False)
        shallow, deep = scores[("얕은하락", AREA)], scores[("깊은하락", AREA)]
        self.assertGreater(shallow["defense"], deep["defense"])   # 방어는 얕은 쪽이 유리
        self.assertGreater(deep["recovery"], shallow["recovery"])  # 회복은 깊은 쪽이 유리
        # 회복 80 : 방어 20이므로 회복 우위가 총점을 뒤집는다.
        self.assertGreater(deep["total"], shallow["total"])
        for score in scores.values():
            self.assertLessEqual(score["defense"], DEFENSE_WEIGHT)
            self.assertLessEqual(score["recovery"], RECOVERY_WEIGHT)
            self.assertEqual(score["peers"], 3)

    def test_never_declined_tops_both_axes(self):
        quarterly = population({"우상향": [100, 110, 120, 130, 140], "하락": [100, 100, 70, 80, 90]})
        scores = build_scores(quarterly, pd.Period("2021Q1", freq="Q"), False)
        steady = scores[("우상향", AREA)]
        self.assertEqual((steady["defense"], steady["recovery"]), (DEFENSE_WEIGHT, RECOVERY_WEIGHT))
        self.assertEqual(steady["total"], 100)
        self.assertGreater(steady["total"], scores[("하락", AREA)]["total"])

    def test_ranking_is_per_area_group(self):
        base = population({"a": [100, 100, 80, 90, 95], "b": [100, 100, 90, 95, 130]})
        other = base.copy()
        other["area_group"] = AREA_LABELS[3]
        other["complex_id"] = other["complex_id"] + "-대형"
        scores = build_scores(pd.concat([base, other], ignore_index=True), pd.Period("2021Q1", freq="Q"), False)
        self.assertEqual(scores[("a", AREA)]["peers"], 2)
        self.assertEqual(scores[("a", AREA)]["total"], scores[("a-대형", AREA_LABELS[3])]["total"])

    def test_rank_reads_against_the_area_group_the_note_names(self):
        # 최하위가 "상위 100%"로 읽히지 않도록 카드는 백분위 대신 순위를 쓴다.
        quarterly = population({"1등": [100, 100, 90, 110, 140],
                                "2등": [100, 100, 80, 90, 105],
                                "꼴찌": [100, 100, 70, 75, 80]})
        scores = build_scores(quarterly, pd.Period("2021Q1", freq="Q"), False)
        self.assertEqual([scores[(name, AREA)]["rank"] for name in ["1등", "2등", "꼴찌"]], [1, 2, 3])
        self.assertEqual({scores[(name, AREA)]["peers"] for name in ["1등", "2등", "꼴찌"]}, {3})

    def test_short_history_is_not_graded(self):
        # 완결 분기 4개를 못 채우면 화면이 자료 부족을 띄우므로 등급도 매기지 않는다.
        quarterly = population({"짧은이력": [100, 90, 95]})
        self.assertEqual(build_scores(quarterly, pd.Period("2020Q3", freq="Q"), False), {})

    def test_explanation_preserves_unrounded_axes_and_tie_percentile(self):
        quarterly = population({"a": [100, 100, 80, 90, 95],
                                "b": [100, 100, 80, 90, 95],
                                "c": [100, 100, 90, 100, 130]})
        scores = build_scores(quarterly, pd.Period("2021Q1", freq="Q"), False)
        a = scores[("a", AREA)]
        self.assertAlmostEqual(a["defense_percentile"], 100 / 3)
        self.assertAlmostEqual(a["recovery_percentile"], 100 / 3)
        self.assertAlmostEqual(a["percentile"], 100 / 3)
        self.assertEqual(a["rank"], 2)
        self.assertEqual(a["axis_peers"], 3)
        self.assertEqual(a["grade"], "C+")
        self.assertEqual(a, scores[("b", AREA)])

    def test_recovery_max_comes_from_all_eligible_area_groups(self):
        base = population({"unit": [100, 100, 80, 90, 100]})
        other = population({"other": [100, 100, 99, 100, 130]})
        other["area_group"] = AREA_LABELS[3]
        insufficient = population({"short": [100, 99.99, 99999]})
        scores = build_scores(pd.concat([base, other, insufficient]), pd.Period("2021Q1", freq="Q"), False)
        for score in scores.values():
            self.assertAlmostEqual(score["recovery_max"], 3100)
            self.assertEqual(score["recovery_sample_count"], 2)
            self.assertEqual(score["peers"], 1)

    def test_no_decline_excluded_from_axes_but_included_in_total_rank(self):
        quarterly = population({"rising": [100, 110, 120, 130, 140],
                                "falling": [100, 100, 80, 90, 100]})
        scores = build_scores(quarterly, pd.Period("2021Q1", freq="Q"), False)
        self.assertTrue(scores[("rising", AREA)]["no_decline"])
        fallen = scores[("falling", AREA)]
        self.assertFalse(fallen["no_decline"])
        self.assertEqual((fallen["axis_peers"], fallen["peers"]), (1, 2))
        self.assertEqual(fallen["defense_percentile"], 50)
        self.assertEqual(fallen["recovery_sample_count"], 1)

    def test_scoring_is_chronological_and_shares_snapshot_basis(self):
        data = fixture([100, 120, 80, 100, 130, 999], counts=[3, 3, 3, 3, 3, 1])
        scores = build_scores(data.quarterly.sample(frac=1, random_state=7), data.reference, False)
        snapshot = build_snapshot(replace(data, scores=scores), "unit", AREA)
        self.assertEqual(snapshot.basis, pd.Period("2021Q1", freq="Q"))
        self.assertEqual(snapshot.score["recovery_max"], snapshot.metrics["recovery"].value)


class ScoreDisclosureTests(unittest.TestCase):
    def snapshot(self, prices):
        data = fixture(prices)
        data.scores = build_scores(data.quarterly, data.reference, data.partial)
        return build_snapshot(data, "unit", AREA)

    def test_disclosure_shows_actual_counts_grade_and_derived_maximum(self):
        s = self.snapshot([100, 100, 99, 100, 130])
        markup = score_details(s)
        self.assertIn('aria-label="등급 산출 기준"', markup)
        self.assertIn(f'{s.score["total"]}점은 이렇게 계산했어요', markup)
        self.assertIn('같은 면적군 1곳 중 <b>1위</b>', markup)
        self.assertIn('3,100.00%', markup)
        self.assertIn('50.0백분위', markup)
        self.assertIn('등급 구간, 현재 B', markup)
        self.assertIn('2021년 1분기', markup)
        self.assertIn('자치구로 제한하지 않아요', markup)

    def test_no_decline_and_insufficient_data_never_invent_a_ratio(self):
        rising = score_details(self.snapshot([100, 110, 120, 130, 140]))
        self.assertIn('회복 점수를 만점', rising)
        self.assertIn('하락분 회복률을 계산할 하락 구간이 없어요', rising)
        self.assertNotIn('nan', rising.lower())
        missing = score_details(self.snapshot([100, 90, 110]))
        self.assertIn('기록이 부족해요', missing)
        self.assertNotIn('백분위', missing)

    def test_demo_does_not_present_seed_scores_as_real_rankings(self):
        from housing.demo import demo_dataset
        s = build_snapshot(demo_dataset(), "demo:리센츠", AREA)
        markup = score_details(s)
        self.assertIn('시연용 점수', markup)
        self.assertNotIn('grade-scale', markup)

    def test_context_is_escaped(self):
        s = self.snapshot([100, 100, 80, 90, 110])
        s.info["district"] = '<script>alert("x")</script>'
        markup = score_details(s)
        self.assertNotIn('<script>', markup)
        self.assertIn('&lt;script&gt;', markup)


class DurationTests(unittest.TestCase):
    def test_quarters_read_as_years_and_months(self):
        self.assertEqual(duration(7), "1년 9개월")
        self.assertEqual(duration(4), "1년")
        self.assertEqual(duration(2), "6개월")
        self.assertEqual(duration(0), "0개월")
        self.assertEqual(duration(None), "자료 부족")

    def test_summary_row_states_why_a_span_is_missing(self):
        recovered = build_snapshot(fixture([100, 120, 80, 100, 130]), "unit", AREA)
        self.assertEqual(recovery_span(recovered.metrics["recovery_quarters"]), "6개월")
        stuck = build_snapshot(fixture([100, 120, 80, 90, 95]), "unit", AREA)
        self.assertEqual(recovery_span(stuck.metrics["recovery_quarters"]), "아직 미회복")
        rising = build_snapshot(fixture([100, 110, 120, 130, 140]), "unit", AREA)
        self.assertEqual(recovery_span(rising.metrics["recovery_quarters"]), "하락 구간 없음")


class SnapshotWiringTests(unittest.TestCase):
    def test_snapshot_reads_the_score_for_its_own_area_group(self):
        data = fixture([100, 120, 80, 100, 130])
        graded = replace(data, scores=build_scores(data.quarterly, data.reference, data.partial))
        self.assertIsNotNone(build_snapshot(graded, "unit", AREA).score)
        # 같은 단지라도 지표가 없는 다른 면적군에는 등급이 붙지 않는다.
        self.assertIsNone(build_snapshot(graded, "unit", AREA_LABELS[0]).score)


if __name__ == "__main__":
    unittest.main()
