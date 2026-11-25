"""Ajuste Dixon-Coles con decaimiento temporal y regularizacion ridge.

Modelo por liga:

    lambda_local = exp(mu + ventaja_local + ataque[local] - defensa[visitante])
    lambda_visita = exp(mu + ataque[visitante] - defensa[local])
    P(x, y) = Poisson(x; lambda_local) * Poisson(y; lambda_visita) * tau(x, y)

donde `defensa[i]` es la *debilidad* defensiva del equipo i (mayor = peor).
`tau` corrige la dependencia en marcadores bajos:

    tau(0,0) = 1 - lh*la*rho    tau(1,0) = 1 + la*rho
    tau(0,1) = 1 + lh*rho       tau(1,1) = 1 - rho

Todo el ajuste es determinista: mismos datos + mismos hiperparametros =>
mismos parametros.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from math import exp, isfinite, lgamma, log
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

DEFAULT_HALF_LIFE = 300.0
DEFAULT_L2 = 0.02
MAX_GOALS = 10


@dataclass
class TeamRating:
    attack: float
    defense: float
    matches: int
    last: str = ""


@dataclass
class LeagueModel:
    league: str
    mu: float
    home_adv: float
    rho: float
    teams: dict[str, TeamRating]
    half_life: float = DEFAULT_HALF_LIFE
    l2: float = DEFAULT_L2
    n_matches: int = 0
    data_through: str = ""
    converged: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mu": self.mu,
            "home_adv": self.home_adv,
            "rho": self.rho,
            "half_life": self.half_life,
            "l2": self.l2,
            "n_matches": self.n_matches,
            "data_through": self.data_through,
            "converged": self.converged,
            "teams": [
                [name, round(r.attack, 5), round(r.defense, 5), r.matches]
                for name, r in sorted(self.teams.items())
            ],
        }


def _days(matches: list[dict]) -> np.ndarray:
    return np.array([_date.fromisoformat(m["d"]).toordinal() for m in matches], dtype=float)


def _weights(matches: list[dict], half_life: float) -> np.ndarray:
    days = _days(matches)
    return np.power(0.5, (days.max() - days) / max(half_life, 1.0))


def fit(
    matches: list[dict],
    half_life: float = DEFAULT_HALF_LIFE,
    l2: float = DEFAULT_L2,
    max_iter: int = 200,
) -> LeagueModel:
    """Ajusta el modelo de una liga. `matches` ya filtrado a esa liga."""
    if len(matches) < 30:
        raise ValueError("Se necesitan al menos 30 partidos para ajustar un modelo")

    teams = sorted({m["h"] for m in matches} | {m["a"] for m in matches})
    index = {name: i for i, name in enumerate(teams)}
    n = len(teams)

    hi = np.array([index[m["h"]] for m in matches], dtype=int)
    ai = np.array([index[m["a"]] for m in matches], dtype=int)
    hg = np.array([m["hg"] for m in matches], dtype=float)
    ag = np.array([m["ag"] for m in matches], dtype=float)
    w = _weights(matches, half_life)

    is00 = (hg == 0) & (ag == 0)
    is10 = (hg == 1) & (ag == 0)
    is01 = (hg == 0) & (ag == 1)
    is11 = (hg == 1) & (ag == 1)

    def unpack(params: np.ndarray):
        mu, home, rho = params[0], params[1], params[2]
        attack = params[3 : 3 + n]
        defense = params[3 + n : 3 + 2 * n]
        return mu, home, rho, attack, defense

    def objective(params: np.ndarray):
        mu, home, rho, attack, defense = unpack(params)
        eta_h = mu + home + attack[hi] - defense[ai]
        eta_a = mu + attack[ai] - defense[hi]
        lam_h = np.exp(eta_h)
        lam_a = np.exp(eta_a)

        log_p = hg * eta_h - lam_h + ag * eta_a - lam_a

        tau = np.ones_like(log_p)
        tau[is00] = 1.0 - lam_h[is00] * lam_a[is00] * rho
        tau[is10] = 1.0 + lam_a[is10] * rho
        tau[is01] = 1.0 + lam_h[is01] * rho
        tau[is11] = 1.0 - rho
        tau = np.clip(tau, 1e-10, None)
        log_p = log_p + np.log(tau)

        value = -np.sum(w * log_p) + l2 * (attack @ attack + defense @ defense)

        # Gradiente analitico (rapido y sin ruido numerico).
        g_h = w * (hg - lam_h)
        g_a = w * (ag - lam_a)

        # Correccion tau -> derivadas respecto a lambda y rho.
        d_rho = np.zeros_like(log_p)
        if is00.any():
            idx = np.where(is00)[0]
            inv = w[idx] / tau[idx]
            g_h[idx] += inv * (-lam_a[idx] * rho)
            g_a[idx] += inv * (-lam_h[idx] * rho)
            d_rho[idx] = inv * (-lam_h[idx] * lam_a[idx])
        if is10.any():
            idx = np.where(is10)[0]
            g_a[idx] += w[idx] * rho / tau[idx]
            d_rho[idx] = w[idx] * lam_a[idx] / tau[idx]
        if is01.any():
            idx = np.where(is01)[0]
            g_h[idx] += w[idx] * rho / tau[idx]
            d_rho[idx] = w[idx] * lam_h[idx] / tau[idx]
        if is11.any():
            idx = np.where(is11)[0]
            d_rho[idx] = w[idx] * (-1.0) / tau[idx]

        grad = np.zeros_like(params)
        grad[0] = -(g_h.sum() + g_a.sum())
        grad[1] = -g_h.sum()
        np.add.at(grad, 3 + hi, -g_h)
        np.add.at(grad, 3 + ai, -g_a)
        np.add.at(grad, 3 + n + ai, g_h)
        np.add.at(grad, 3 + n + hi, g_a)
        grad[2] = -d_rho.sum()
        grad[3 : 3 + n] += 2 * l2 * attack
        grad[3 + n : 3 + 2 * n] += 2 * l2 * defense
        return value, grad

    x0 = np.zeros(3 + 2 * n)
    x0[0] = log(max(np.mean(hg + ag) / 2.0, 0.3))
    x0[1] = 0.25
    x0[2] = -0.05
    bounds = [(-2.0, 2.0), (-1.0, 1.0), (-0.4, 0.4)] + [(-3.0, 3.0)] * (2 * n)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-10, "maxls": 40},
    )

    mu, home, rho, attack, defense = unpack(result.x)
    attack = attack - attack.mean()
    defense = defense - defense.mean()
    mu = mu + (result.x[3 : 3 + n].mean() - result.x[3 + n : 3 + 2 * n].mean())

    counts = {name: 0 for name in teams}
    last_played = {name: "" for name in teams}
    for match in matches:
        for side in ("h", "a"):
            name = match[side]
            counts[name] += 1
            if match["d"] > last_played[name]:
                last_played[name] = match["d"]

    outcomes = np.zeros(3)
    for match in matches:
        outcomes[0 if match["hg"] > match["ag"] else (1 if match["hg"] == match["ag"] else 2)] += 1
    base_probs = {
        "home": float(outcomes[0] / outcomes.sum()),
        "draw": float(outcomes[1] / outcomes.sum()),
        "away": float(outcomes[2] / outcomes.sum()),
    }

    return LeagueModel(
        league=matches[0]["l"],
        mu=float(mu),
        home_adv=float(home),
        rho=float(rho),
        teams={
            name: TeamRating(float(attack[i]), float(defense[i]), counts[name], last_played[name])
            for i, name in enumerate(teams)
        },
        half_life=half_life,
        l2=l2,
        n_matches=len(matches),
        data_through=max(m["d"] for m in matches),
        converged=bool(result.success),
        extra={"nll": float(result.fun), "iterations": int(result.nit), "base_probs": base_probs},
    )


def lambdas(model: LeagueModel, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
    if home not in model.teams:
        raise KeyError(f"Equipo desconocido en {model.league}: {home}")
    if away not in model.teams:
        raise KeyError(f"Equipo desconocido en {model.league}: {away}")
    venue = 0.0 if neutral else model.home_adv
    lam_h = exp(model.mu + venue + model.teams[home].attack - model.teams[away].defense)
    lam_a = exp(model.mu + model.teams[away].attack - model.teams[home].defense)
    return float(np.clip(lam_h, 0.05, 8.0)), float(np.clip(lam_a, 0.05, 8.0))


def score_matrix(
    model: LeagueModel, home: str, away: str, neutral: bool = False, max_goals: int = MAX_GOALS
) -> np.ndarray:
    lam_h, lam_a = lambdas(model, home, away, neutral)
    logs = np.array([k * log(lam_h) - lam_h - lgamma(k + 1) for k in range(max_goals + 1)])
    loga = np.array([k * log(lam_a) - lam_a - lgamma(k + 1) for k in range(max_goals + 1)])
    matrix = np.exp(logs[:, None] + loga[None, :])

    # Correccion Dixon-Coles en las cuatro celdas de marcador bajo.
    matrix[0, 0] *= 1.0 - lam_h * lam_a * model.rho
    matrix[1, 0] *= 1.0 + lam_a * model.rho
    matrix[0, 1] *= 1.0 + lam_h * model.rho
    matrix[1, 1] *= 1.0 - model.rho
    matrix = np.clip(matrix, 1e-12, None)
    return matrix / matrix.sum()


def elo_ratings(matches: list[dict], k: float = 20.0, home_bonus: float = 60.0) -> dict[str, float]:
    """Elo secuencial. Se exporta como referencia descriptiva, no como motor."""
    ratings: dict[str, float] = {}
    for match in sorted(matches, key=lambda m: m["d"]):
        home, away = match["h"], match["a"]
        rh = ratings.setdefault(home, 1500.0)
        ra = ratings.setdefault(away, 1500.0)
        expected = 1.0 / (1.0 + 10 ** ((ra - (rh + home_bonus)) / 400.0))
        score = 1.0 if match["hg"] > match["ag"] else (0.5 if match["hg"] == match["ag"] else 0.0)
        delta = k * (score - expected)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def reliability(model: LeagueModel, holdout: list[dict], bins: int = 10) -> list[dict]:
    """Curva de fiabilidad sobre un holdout: prob predicha vs frecuencia observada."""
    counts = np.zeros(bins)
    sums = np.zeros(bins)
    observed = np.zeros(bins)
    for match in holdout:
        if match["h"] not in model.teams or match["a"] not in model.teams:
            continue
        matrix = score_matrix(model, match["h"], match["a"])
        probs = [
            float(np.tril(matrix, -1).sum()),
            float(np.diag(matrix).sum()),
            float(np.triu(matrix, 1).sum()),
        ]
        outcome = 0 if match["hg"] > match["ag"] else (1 if match["hg"] == match["ag"] else 2)
        for cls, prob in enumerate(probs):
            bucket = min(int(prob * bins), bins - 1)
            counts[bucket] += 1
            sums[bucket] += prob
            observed[bucket] += 1.0 if outcome == cls else 0.0
    return [
        {
            "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
            "count": int(counts[i]),
            "predicted": round(float(sums[i] / counts[i]), 4) if counts[i] else None,
            "observed": round(float(observed[i] / counts[i]), 4) if counts[i] else None,
        }
        for i in range(bins)
    ]


def finite_or_none(value: float) -> float | None:
    return float(value) if isfinite(value) else None
