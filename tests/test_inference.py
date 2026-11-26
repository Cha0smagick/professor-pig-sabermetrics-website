import numpy as np
import pytest

from sabermetria import inference


def _preds(rows):
    return [{"home": h, "draw": d, "away": a} for h, d, a in rows]


def test_wilson_bounds():
    low = inference.wilson_interval(0, 10)
    high = inference.wilson_interval(10, 10)
    assert low[0] == 0.0
    assert 0.2 < low[1] < 0.35
    assert 0.65 < high[0] < 0.8
    assert high[1] == 1.0
    assert inference.wilson_interval(0, 0) == (0.0, 0.0)


def test_brier_decomposition_identity():
    rows = [(0.6, 0.25, 0.15), (0.2, 0.3, 0.5), (0.4, 0.4, 0.2), (0.7, 0.2, 0.1)]
    outcomes = [0, 2, 1, 0]
    preds = _preds(rows)
    brier = inference.metric_set(np.array(rows), np.array(outcomes))["brier"]
    parts = inference.brier_decomposition(preds, outcomes, bins=5)
    reconstructed = parts["reliability"] - parts["resolution"] + parts["uncertainty"]
    assert abs(brier - reconstructed) < 0.05


def test_benjamini_hochberg_monotone_and_significant():
    result = inference.benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.4})
    assert result["a"]["significant"] is True
    assert result["d"]["significant"] is False
    assert result["a"]["q"] <= result["b"]["q"] <= result["c"]["q"] <= result["d"]["q"]


def test_block_bootstrap_returns_intervals_and_p_value():
    rng = np.random.default_rng(3)
    rows, outcomes, blocks, base = [], [], [], []
    for block in range(20):
        for _ in range(10):
            probs = rng.dirichlet([2, 2, 2])
            rows.append({"home": float(probs[0]), "draw": float(probs[1]), "away": float(probs[2])})
            outcomes.append(int(rng.integers(0, 3)))
            blocks.append(f"W{block}")
            base.append({"home": 0.45, "draw": 0.25, "away": 0.30})
    result = inference.block_bootstrap(rows, outcomes, blocks, base, n_boot=100)
    assert result["blocks"] == 20
    assert len(result["brier_ci"]) == 2
    assert 0.001 <= result["p_value"] <= 1
    assert result["bss"] is not None


def test_spearman_detects_monotone_trend():
    result = inference.spearman_trend([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["rho"] == pytest.approx(1.0)
    assert result["p"] < 0.05


def test_pearson_perfect_correlation():
    result = inference.pearson_with_ci([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert result["r"] == pytest.approx(1.0)
    assert result["p"] < 0.05


def test_chi_square_identical_distribution():
    result = inference.chi_square_gof([10, 20, 30, 40], [10, 20, 30, 40])
    assert result["chi2"] == pytest.approx(0.0)
    assert result["p"] == pytest.approx(1.0)


def test_dispersion_index():
    poisson_like = inference.dispersion_index([1, 0, 2, 1, 1, 0, 2, 1])
    assert poisson_like["index"] > 0
    constant = inference.dispersion_index([1] * 10)
    assert constant["index"] == 0.0
