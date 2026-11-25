"""Mercados derivados de la matriz de marcadores. Funciones puras y testeables."""

from __future__ import annotations

import numpy as np

OUTCOMES = ("home", "draw", "away")
FINITE = 1e-12


def _clip(value: float) -> float:
    return float(min(max(value, FINITE), 1.0 - FINITE))


def outcome_probs(matrix: np.ndarray) -> dict[str, float]:
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.diag(matrix).sum())
    away = float(np.triu(matrix, 1).sum())
    total = home + draw + away
    return {"home": home / total, "draw": draw / total, "away": away / total}


def top_scores(matrix: np.ndarray, n: int = 5) -> list[dict]:
    flat = np.argsort(matrix, axis=None)[::-1][:n]
    return [
        {"home": int(i // matrix.shape[1]), "away": int(i % matrix.shape[1]), "p": float(matrix.flat[i])}
        for i in flat
    ]


def double_chance(probs: dict[str, float]) -> dict[str, float]:
    return {
        "1X": probs["home"] + probs["draw"],
        "12": probs["home"] + probs["away"],
        "X2": probs["draw"] + probs["away"],
    }


def over_under(matrix: np.ndarray, lines: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5)) -> list[dict]:
    goals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    return [
        {"line": line, "over": float(matrix[goals > line].sum()), "under": float(matrix[goals < line].sum())}
        for line in lines
    ]


def btts(matrix: np.ndarray) -> float:
    # P(local>=1 y visitante>=1)
    return float(1.0 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0])


def asian_handicap(matrix: np.ndarray, line: float) -> dict[str, float]:
    """Handicap asiatico desde el punto de vista del local. `line` se suma al local."""
    goals = np.subtract.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1])) + line
    win = float(matrix[goals > 0].sum())
    push = float(matrix[goals == 0].sum())
    lose = float(matrix[goals < 0].sum())
    return {"win": win, "push": push, "lose": lose}


def expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    home = np.arange(matrix.shape[0])[:, None]
    away = np.arange(matrix.shape[1])[None, :]
    return float((matrix * home).sum()), float((matrix * away).sum())


def confidence(probs: dict[str, float]) -> float:
    """1 - entropia normalizada. 0 = total incertidumbre, 1 = certeza."""
    p = np.array([probs[k] for k in OUTCOMES], dtype=float)
    p = np.clip(p, FINITE, 1.0)
    entropy = -float((p * np.log(p)).sum()) / np.log(len(p))
    return float(min(max(1.0 - entropy, 0.0), 1.0))


def value_bets(probs: dict[str, float], odds: dict[str, float], kelly_fraction: float = 0.25, cap: float = 0.05) -> list[dict]:
    bets = []
    for outcome in OUTCOMES:
        price = odds.get(outcome)
        if not price or price <= 1.0:
            continue
        p = probs[outcome]
        ev = p * price - 1.0
        kelly = max(0.0, ev / (price - 1.0)) * kelly_fraction
        bets.append(
            {
                "outcome": outcome,
                "probability": p,
                "odds": price,
                "fair_odds": 1.0 / p,
                "edge": ev,
                "kelly": min(kelly, cap),
                "positive": ev > 0,
            }
        )
    return bets


def full_report(matrix: np.ndarray, odds: dict[str, float] | None = None) -> dict:
    probs = outcome_probs(matrix)
    xg_home, xg_away = expected_goals(matrix)
    report = {
        "probabilities": probs,
        "expected_goals": {"home": xg_home, "away": xg_away},
        "double_chance": double_chance(probs),
        "top_scores": top_scores(matrix),
        "over_under": over_under(matrix),
        "btts": btts(matrix),
        "handicap": {str(line): asian_handicap(matrix, line) for line in (-1.0, -0.5, 0.5, 1.0)},
        "confidence": confidence(probs),
        "fair_odds": {k: 1.0 / v for k, v in probs.items()},
    }
    if odds:
        report["value"] = value_bets(probs, odds)
    return report
