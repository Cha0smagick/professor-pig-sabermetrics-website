"""Inferencia estadística: incertidumbre, comparaciones múltiples y calibración.

Ninguna cifra inferencial se calcula con errores estándar i.i.d. ingenuos:
los partidos se agrupan en bloques semanales dentro de liga y temporada, y el
bootstrap remuestrea bloques completos.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy import stats as sps

BINS = 10


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _as_arrays(preds, outcomes):
    probs = np.asarray([[p["home"], p["draw"], p["away"]] for p in preds], dtype=float)
    obs = np.asarray(outcomes, dtype=int)
    probs = np.clip(probs, 1e-12, 1.0)
    return probs, obs


def metric_set(probs: np.ndarray, obs: np.ndarray) -> dict:
    onehot = np.eye(3)[obs]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    logloss = float(-np.mean(np.log(probs[np.arange(len(obs)), obs])))
    accuracy = float(np.mean(np.argmax(probs, axis=1) == obs))
    return {"n": int(len(obs)), "brier": brier, "logloss": logloss, "accuracy": accuracy}


def brier_decomposition(preds, outcomes, bins: int = BINS) -> dict:
    """Murphy para multiclase: se descompone cada clase y se suma.

    Brier = reliability - resolution + uncertainty. Menor reliability y mayor
    resolution son mejores. Se documenta como suma de las tres clases.
    """
    probs, obs = _as_arrays(preds, outcomes)
    reliability = resolution = uncertainty = 0.0
    for cls in range(3):
        p = probs[:, cls]
        y = (obs == cls).astype(float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        bucket = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
        base = y.mean()
        uncertainty += base * (1 - base)
        for b in range(bins):
            mask = bucket == b
            count = int(mask.sum())
            if not count:
                continue
            p_bar = float(p[mask].mean())
            y_bar = float(y[mask].mean())
            reliability += count / len(p) * (p_bar - y_bar) ** 2
            resolution += count / len(p) * (y_bar - base) ** 2
    return {
        "reliability": round(float(reliability), 5),
        "resolution": round(float(resolution), 5),
        "uncertainty": round(float(uncertainty), 5),
    }


def expected_calibration_error(preds, outcomes, bins: int = BINS) -> float:
    probs, obs = _as_arrays(preds, outcomes)
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for cls in range(3):
        p = probs[:, cls]
        y = (obs == cls).astype(float)
        bucket = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
        for b in range(bins):
            mask = bucket == b
            count = int(mask.sum())
            if count:
                total += count / len(p) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total / 3)


def block_bootstrap(
    preds,
    outcomes,
    blocks: list[str],
    base_preds=None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """IC percentiles y p-valor pareado remuestreando bloques completos."""
    probs, obs = _as_arrays(preds, outcomes)
    base = _as_arrays(base_preds, outcomes)[0] if base_preds is not None else None
    block_array = np.asarray(blocks)
    unique = np.unique(block_array)
    index = {block: np.flatnonzero(block_array == block) for block in unique}
    rng = np.random.default_rng(seed)

    brier, logloss, accuracy, delta, bss = [], [], [], [], []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[block] for block in chosen])
        sample = metric_set(probs[idx], obs[idx])
        brier.append(sample["brier"])
        logloss.append(sample["logloss"])
        accuracy.append(sample["accuracy"])
        if base is not None:
            base_sample = metric_set(base[idx], obs[idx])
            delta.append(base_sample["brier"] - sample["brier"])
            if base_sample["brier"] > 1e-12:
                bss.append(1 - sample["brier"] / base_sample["brier"])

    def interval(values, digits=5):
        if not values:
            return None
        return [round(float(np.percentile(values, 2.5)), digits), round(float(np.percentile(values, 97.5)), digits)]

    result = {
        "blocks": int(len(unique)),
        "brier_ci": interval(brier),
        "logloss_ci": interval(logloss),
        "accuracy_ci": interval(accuracy),
    }
    if delta:
        delta_array = np.asarray(delta)
        # p unilateral con corrección +1 para evitar el suelo en 0.
        p_value = (float(np.sum(delta_array <= 0)) + 1.0) / (n_boot + 1.0)
        result.update(
            {
                "delta_brier": round(float(np.mean(delta_array)), 5),
                "delta_brier_ci": interval(delta),
                "p_value": round(min(1.0, p_value), 4),
                "bss": round(float(np.mean(bss)), 5) if bss else None,
                "bss_ci": interval(bss) if bss else None,
            }
        )
    return result


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Control de falsos descubrimientos sobre las ligas comparadas."""
    items = sorted(((name, p) for name, p in p_values.items() if p is not None), key=lambda item: item[1])
    m = len(items)
    output: dict[str, dict] = {}
    previous = 1.0
    for rank in range(m, 0, -1):
        name, p = items[rank - 1]
        q = min(previous, p * m / rank)
        previous = q
        output[name] = {"p": round(float(p), 4), "q": round(float(q), 4), "significant": q <= alpha}
    for name, p in p_values.items():
        if name not in output:
            output[name] = {"p": p, "q": None, "significant": False}
    return output


def spearman_trend(values: list[float]) -> dict:
    if len(values) < 4 or float(np.ptp(np.asarray(values, dtype=float))) == 0.0:
        return {"rho": None, "p": None, "n": len(values)}
    x = np.arange(len(values), dtype=float)
    rho, p = sps.spearmanr(x, np.asarray(values, dtype=float))
    return {"rho": round(float(rho), 3), "p": round(float(p), 4), "n": len(values)}


def pearson_with_ci(x: list[float], y: list[float]) -> dict:
    n = len(x)
    if n < 5:
        return {"r": None, "p": None, "ci": None, "n": n}
    r, p = sps.pearsonr(x, y)
    if abs(r) >= 1:
        ci = [r, r]
    else:
        z = np.arctanh(r)
        half = 1.96 / sqrt(n - 3)
        ci = [round(float(np.tanh(z - half)), 3), round(float(np.tanh(z + half)), 3)]
    return {"r": round(float(r), 3), "p": round(float(p), 4), "ci": ci, "n": n}


def chi_square_gof(observed: list[int], expected: list[float]) -> dict:
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    mask = exp > 0
    obs, exp = obs[mask], exp[mask]
    if len(obs) < 3 or exp.sum() <= 0:
        return {"chi2": None, "df": None, "p": None}
    scale = obs.sum() / exp.sum()
    exp = exp * scale
    # agrupar colas para que la esperanza mínima sea 5
    while len(exp) > 3 and exp.min() < 5:
        idx = int(np.argmin(exp))
        if idx == 0:
            exp[1] += exp[0]; obs[1] += obs[0]
        else:
            exp[idx - 1] += exp[idx]; obs[idx - 1] += obs[idx]
        obs = np.delete(obs, idx); exp = np.delete(exp, idx)
    statistic = float(((obs - exp) ** 2 / exp).sum())
    df = len(obs) - 1
    return {"chi2": round(statistic, 3), "df": df, "p": round(float(sps.chi2.sf(statistic, df)), 4)}


def dispersion_index(goals: list[int]) -> dict:
    values = np.asarray(goals, dtype=float)
    mean = float(values.mean())
    if mean <= 0:
        return {"index": None, "mean": 0.0, "variance": 0.0}
    variance = float(values.var(ddof=1))
    return {
        "index": round(variance / mean, 3),
        "mean": round(mean, 3),
        "variance": round(variance, 3),
    }
