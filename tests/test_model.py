import math

import numpy as np
import pytest

from sabermetria import model as engine


def simple_matrix(lam_h=1.5, lam_a=1.0, size=8):
    ph = [np.exp(-lam_h) * lam_h**k / math.factorial(k) for k in range(size)]
    pa = [np.exp(-lam_a) * lam_a**k / math.factorial(k) for k in range(size)]
    matrix = np.array([[x * y for y in pa] for x in ph])
    return matrix / matrix.sum()


def test_outcome_probs_sum_to_one():
    probs = engine.outcome_probs(simple_matrix())
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0 < value < 1 for value in probs.values())


def test_stronger_home_team_has_higher_home_probability():
    weak = engine.outcome_probs(simple_matrix(1.0, 1.5))
    strong = engine.outcome_probs(simple_matrix(2.0, 0.8))
    assert strong["home"] > weak["home"]


def test_over_under_are_complementary_and_monotonic():
    matrix = simple_matrix()
    rows = engine.over_under(matrix)
    for row in rows:
        assert row["over"] + row["under"] == pytest.approx(1.0)
    overs = [row["over"] for row in rows]
    assert overs == sorted(overs, reverse=True)


def test_btts_bounds():
    matrix = simple_matrix()
    value = engine.btts(matrix)
    assert 0 < value < 1


def test_top_scores_are_sorted_and_normalized():
    matrix = simple_matrix()
    scores = engine.top_scores(matrix, 5)
    assert len(scores) == 5
    probs = [item["p"] for item in scores]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0


def test_handicap_outcomes_partition_the_mass():
    matrix = simple_matrix()
    for line in (-1.0, -0.5, 0.5, 1.0):
        result = engine.asian_handicap(matrix, line)
        assert result["win"] + result["push"] + result["lose"] == pytest.approx(1.0)


def test_kelly_is_zero_without_edge():
    report = engine.value_bets({"home": 0.5, "draw": 0.3, "away": 0.2}, {"home": 2.0})
    assert report[0]["edge"] == pytest.approx(0.0)
    assert report[0]["kelly"] == 0.0


def test_value_detects_positive_edge_and_caps_kelly():
    report = engine.value_bets({"home": 0.6, "draw": 0.25, "away": 0.15}, {"home": 2.0})
    assert report[0]["edge"] == pytest.approx(0.2)
    assert report[0]["positive"] is True
    assert report[0]["kelly"] <= 0.05


def test_confidence_between_zero_and_one():
    certain = engine.confidence({"home": 0.99, "draw": 0.005, "away": 0.005})
    uncertain = engine.confidence({"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3})
    assert 0.0 <= uncertain < certain <= 1.0


def test_full_report_contains_all_markets():
    report = engine.full_report(simple_matrix(), {"home": 2.1, "draw": 3.4, "away": 3.6})
    for key in ("probabilities", "expected_goals", "double_chance", "top_scores", "over_under", "btts", "handicap", "confidence", "fair_odds", "value"):
        assert key in report
