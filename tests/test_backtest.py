import pytest

from sabermetria import backtest


def test_metrics_perfect_prediction():
    metrics = backtest._metrics([{"home": 1.0, "draw": 0.0, "away": 0.0}], [0])
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["logloss"] == pytest.approx(0.0)
    assert metrics["accuracy"] == 1.0


def test_metrics_worst_prediction():
    metrics = backtest._metrics([{"home": 0.0, "draw": 0.0, "away": 1.0}], [0])
    assert metrics["brier"] == pytest.approx(2.0)
    assert metrics["accuracy"] == 0.0


def test_empty_metrics_are_none():
    metrics = backtest._metrics([], [])
    assert metrics["n"] == 0
    assert metrics["brier"] is None


def test_baseline_is_a_distribution():
    train = [
        {"hg": 2, "ag": 0},
        {"hg": 1, "ag": 1},
        {"hg": 0, "ag": 3},
        {"hg": 2, "ag": 1},
    ]
    base = backtest._baseline(train)
    assert sum(base.values()) == pytest.approx(1.0)
    assert base["home"] == pytest.approx(0.5)


def test_ordered_seasons_uses_dates_not_strings():
    matches = [
        {"s": "2025", "d": "2025-02-01"},
        {"s": "2019-20", "d": "2019-08-01"},
        {"s": "2020", "d": "2020-03-01"},
    ]
    assert backtest.ordered_seasons(matches) == ["2019-20", "2020", "2025"]


def test_alpha_blend_moves_towards_baseline():
    from sabermetria import ratings

    matches = []
    for i in range(40):
        matches.append({"l": "en.1", "s": "2024-25", "d": f"2024-08-{1 + i % 28:02d}", "h": "A", "a": "B", "hg": 2, "ag": 0})
        matches.append({"l": "en.1", "s": "2024-25", "d": f"2024-09-{1 + i % 28:02d}", "h": "B", "a": "A", "hg": 1, "ag": 1})
    model = ratings.fit(matches)
    base = {"home": 0.9, "draw": 0.05, "away": 0.05}
    raw, _ = backtest._predict(model, [{"h": "A", "a": "B", "hg": 1, "ag": 0}], base, 0.0)
    blended, _ = backtest._predict(model, [{"h": "A", "a": "B", "hg": 1, "ag": 0}], base, 0.5)
    assert blended[0]["home"] > raw[0]["home"]
