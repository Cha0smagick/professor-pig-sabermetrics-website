import numpy as np
import pytest

from sabermetria import model as engine
from sabermetria import ratings


def synthetic_matches(n_teams: int = 6, rounds: int = 60, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    day = 1
    for _ in range(rounds):
        for i in range(n_teams):
            home = teams[i]
            away = teams[(i + 1 + _ % (n_teams - 1)) % n_teams]
            if home == away:
                continue
            lam_h = 1.6 + 0.3 * (i - n_teams / 2) * 0.2
            lam_a = 1.1
            rows.append(
                {
                    "l": "en.1",
                    "s": "2024-25",
                    "d": f"2024-{1 + (day // 28) % 12:02d}-{1 + day % 28:02d}",
                    "h": home,
                    "a": away,
                    "hg": int(rng.poisson(lam_h)),
                    "ag": int(rng.poisson(lam_a)),
                }
            )
            day += 1
    return rows


@pytest.fixture(scope="module")
def model():
    return ratings.fit(synthetic_matches(), half_life=300, l2=0.02)


def test_matrix_is_stochastic(model):
    matrix = ratings.score_matrix(model, "T0", "T1")
    assert matrix.min() >= 0
    assert abs(matrix.sum() - 1.0) < 1e-9


def test_lambdas_positive(model):
    lam_h, lam_a = ratings.lambdas(model, "T0", "T1")
    assert 0 < lam_h <= 8
    assert 0 < lam_a <= 8


def test_neutral_venue_matches_transposed_matrix(model):
    model.home_adv = 0.0
    direct = ratings.score_matrix(model, "T0", "T1")
    transposed = ratings.score_matrix(model, "T1", "T0").T
    assert np.max(np.abs(direct - transposed)) < 1e-12


def test_home_advantage_increases_home_goals(model):
    model.home_adv = 0.25
    neutral = ratings.score_matrix(model, "T0", "T1", neutral=True)
    home = ratings.score_matrix(model, "T0", "T1", neutral=False)
    xg_neutral = engine.outcome_probs(neutral)["home"]
    xg_home = engine.outcome_probs(home)["home"]
    assert xg_home > xg_neutral


def test_fit_is_reproducible():
    matches = synthetic_matches()
    first = ratings.fit(matches)
    second = ratings.fit(matches)
    assert first.rho == pytest.approx(second.rho, abs=0)
    assert first.mu == pytest.approx(second.mu, abs=0)
    assert first.home_adv == pytest.approx(second.home_adv, abs=0)
    assert [r.attack for r in first.teams.values()] == [r.attack for r in second.teams.values()]


def test_centered_ratings(model):
    attacks = [r.attack for r in model.teams.values()]
    defenses = [r.defense for r in model.teams.values()]
    assert abs(np.mean(attacks)) < 1e-9
    assert abs(np.mean(defenses)) < 1e-9


def test_unknown_team_raises(model):
    with pytest.raises(KeyError):
        ratings.lambdas(model, "T0", "NO EXISTE")


def test_extreme_rho_never_produces_negative_probabilities(model):
    model.rho = 0.5
    matrix = ratings.score_matrix(model, "T0", "T1")
    assert matrix.min() > 0
    assert abs(matrix.sum() - 1.0) < 1e-9


def test_too_few_matches_rejected():
    with pytest.raises(ValueError):
        ratings.fit(synthetic_matches(rounds=1))


def test_elo_is_zero_sum_per_match():
    matches = synthetic_matches(rounds=3)
    elo = ratings.elo_ratings(matches)
    assert len(elo) == 6
    before = sum(elo.values())
    assert abs(before - 9000.0) < 1e-6
