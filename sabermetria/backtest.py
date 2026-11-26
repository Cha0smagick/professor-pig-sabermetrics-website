"""Validacion walk-forward: sin fuga temporal y contra baselines explicitos.

Baselines:
- `freq`: frecuencias de victoria local/empate/visita del propio entrenamiento.
- `home`: predecir siempre victoria local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date

import numpy as np

from . import inference, ratings
from .model import outcome_probs

DEFAULT_GRID = ((180.0, 0.02), (300.0, 0.02), (420.0, 0.02), (300.0, 0.05))
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
LABELS = ("home", "draw", "away")


def _ordinal(value: str) -> int:
    return _date.fromisoformat(value).toordinal()


def ordered_seasons(matches: list[dict]) -> list[str]:
    first: dict[str, int] = {}
    for match in matches:
        key = match["s"]
        value = _ordinal(match["d"])
        if key not in first or value < first[key]:
            first[key] = value
    return [season for season, _ in sorted(first.items(), key=lambda item: item[1])]


def _outcome(match: dict) -> int:
    if match["hg"] > match["ag"]:
        return 0
    if match["hg"] == match["ag"]:
        return 1
    return 2


def _block(match: dict) -> str:
    week = _date.fromisoformat(match["d"]).isocalendar().week
    return f"{match['s']}-W{week:02d}"


def _metrics(preds: list[dict], outcomes: list[int]) -> dict:
    if not preds:
        return {"n": 0, "brier": None, "logloss": None, "accuracy": None}
    probs = np.array([[p["home"], p["draw"], p["away"]] for p in preds], dtype=float)
    obs = np.array(outcomes, dtype=int)
    probs = np.clip(probs, 1e-12, 1.0)
    onehot = np.eye(3)[obs]
    return {
        "n": int(len(obs)),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "logloss": float(-np.mean(np.log(probs[np.arange(len(obs)), obs]))),
        "accuracy": float(np.mean(np.argmax(probs, axis=1) == obs)),
    }


def _baseline(train: list[dict]) -> dict[str, float]:
    counts = np.zeros(3)
    for match in train:
        counts[_outcome(match)] += 1
    if counts.sum() == 0:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    counts = counts / counts.sum()
    return {"home": float(counts[0]), "draw": float(counts[1]), "away": float(counts[2])}


def tune(
    matches: list[dict],
    grid=DEFAULT_GRID,
    holdout_season: str | None = None,
) -> tuple[float, float, float, list[dict]]:
    """Elige (semivida, ridge, mezcla con baseline) en una temporada pasada.

    `evaluate_league` pasa explícitamente una temporada anterior al periodo de
    test, de modo que la temporada de ajuste nunca se reutiliza como test.
    """
    seasons = ordered_seasons(matches)
    if len(seasons) < 3:
        return ratings.DEFAULT_HALF_LIFE, ratings.DEFAULT_L2, 0.0, []
    if holdout_season is None:
        holdout_season = seasons[-2]
    if holdout_season not in seasons:
        return ratings.DEFAULT_HALF_LIFE, ratings.DEFAULT_L2, 0.0, []
    start = min(_ordinal(m["d"]) for m in matches if m["s"] == holdout_season)
    train = [m for m in matches if _ordinal(m["d"]) < start]
    test = [m for m in matches if m["s"] == holdout_season]
    if len(train) < 30 or not test:
        return ratings.DEFAULT_HALF_LIFE, ratings.DEFAULT_L2, 0.0, []

    trials, models = [], {}
    for half_life, l2 in grid:
        model = ratings.fit(train, half_life=half_life, l2=l2)
        preds, outcomes, _ = _predict(model, test)
        score = _metrics(preds, outcomes)
        trials.append(
            {
                "half_life": half_life,
                "l2": l2,
                "logloss": score["logloss"],
                "brier": score["brier"],
                "n": score["n"],
            }
        )
        models[(half_life, l2)] = model

    valid = [t for t in trials if t["logloss"] is not None]
    if not valid:
        return ratings.DEFAULT_HALF_LIFE, ratings.DEFAULT_L2, 0.0, trials
    best = min(valid, key=lambda t: t["logloss"])
    model = models[(best["half_life"], best["l2"])]
    base = model.extra["base_probs"]

    alpha, alpha_logloss = 0.0, best["logloss"]
    for candidate in ALPHAS:
        preds, outcomes, _ = _predict(model, test, base, candidate)
        score = _metrics(preds, outcomes)
        trials.append(
            {"half_life": best["half_life"], "l2": best["l2"], "alpha": candidate,
             "logloss": score["logloss"], "brier": score["brier"], "n": score["n"]}
        )
        if score["logloss"] is not None and score["logloss"] < alpha_logloss - 1e-6:
            alpha, alpha_logloss = candidate, score["logloss"]

    return best["half_life"], best["l2"], alpha, trials


def _predict(
    model: ratings.LeagueModel,
    test: list[dict],
    base: dict[str, float] | None = None,
    alpha: float = 0.0,
) -> tuple[list[dict], list[int], list[str]]:
    preds, outcomes, blocks = [], [], []
    for match in test:
        if match["h"] not in model.teams or match["a"] not in model.teams:
            continue
        matrix = ratings.score_matrix(model, match["h"], match["a"])
        probs = outcome_probs(matrix)
        if base and alpha > 0:
            probs = {k: (1 - alpha) * probs[k] + alpha * base[k] for k in probs}
        preds.append(probs)
        outcomes.append(_outcome(match))
        blocks.append(_block(match))
    return preds, outcomes, blocks


def _reliability(preds: list[dict], outcomes: list[int], bins: int = 10) -> list[dict]:
    counts = np.zeros(bins)
    sums = np.zeros(bins)
    observed = np.zeros(bins)
    for probs, outcome in zip(preds, outcomes):
        for cls in range(3):
            prob = probs[LABELS[cls]]
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


@dataclass
class LeagueResult:
    model: ratings.LeagueModel
    status: str
    metrics: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    reliability: list = field(default_factory=list)
    tuning: list = field(default_factory=list)
    coverage: float = 0.0
    preds: list = field(default_factory=list)
    base_preds: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    keys: list = field(default_factory=list)
    inference: dict = field(default_factory=dict)


def evaluate_league(matches: list[dict], tune_first: bool = True) -> LeagueResult:
    seasons = ordered_seasons(matches)
    if len(seasons) >= 8:
        test_count = 5
    elif len(seasons) >= 5:
        test_count = 3
    else:
        test_count = 1
    test_start = max(0, len(seasons) - test_count)
    test_seasons = seasons[test_start:]
    holdout = seasons[test_start - 1] if test_start >= 2 else None

    if tune_first and holdout is not None:
        half_life, l2, alpha, trials = tune(matches, holdout_season=holdout)
    elif tune_first:
        half_life, l2, alpha, trials = tune(matches)
    else:
        half_life, l2, alpha, trials = ratings.DEFAULT_HALF_LIFE, ratings.DEFAULT_L2, 0.0, []

    preds, outcomes, blocks, keys, base_preds = [], [], [], [], []
    total = 0
    for season in test_seasons:
        start = min(_ordinal(m["d"]) for m in matches if m["s"] == season)
        train = [m for m in matches if _ordinal(m["d"]) < start]
        test = [m for m in matches if m["s"] == season]
        if len(train) < 30:
            continue
        model = ratings.fit(train, half_life=half_life, l2=l2)
        base = _baseline(train)
        season_preds, season_outcomes, season_blocks = _predict(model, test, base, alpha)
        season_keys = [
            (m["d"], m["h"], m["a"])
            for m in test
            if m["h"] in model.teams and m["a"] in model.teams
        ]
        preds.extend(season_preds)
        outcomes.extend(season_outcomes)
        blocks.extend(season_blocks)
        keys.extend(season_keys)
        base_preds.extend([base] * len(season_outcomes))
        total += len(test)

    metrics = _metrics(preds, outcomes)
    baseline = _metrics(base_preds, outcomes) if base_preds else {"n": 0}
    home_hits = sum(1 for o in outcomes if o == 0)
    if outcomes:
        baseline["home_accuracy"] = home_hits / len(outcomes)

    if len(matches) >= 120 and len(seasons) >= 2:
        status = "predictable"
    elif len(matches) >= 60:
        status = "limited"
    else:
        status = "insufficient"

    final_model = ratings.fit(matches, half_life=half_life, l2=l2)
    final_model.extra["alpha"] = alpha
    rel = _reliability(preds, outcomes)

    interval = {}
    if preds:
        interval = inference.block_bootstrap(preds, outcomes, blocks, base_preds)
        interval["brier_decomposition"] = inference.brier_decomposition(preds, outcomes)
        interval["ece"] = round(inference.expected_calibration_error(preds, outcomes), 5)
        interval["test_seasons"] = test_seasons

    return LeagueResult(
        model=final_model,
        status=status,
        metrics=metrics,
        baseline=baseline,
        reliability=rel,
        tuning=trials,
        coverage=(len(preds) / total) if total else 0.0,
        preds=preds,
        base_preds=base_preds,
        outcomes=outcomes,
        blocks=blocks,
        keys=keys,
        inference=interval,
    )


def aggregate(results: dict[str, LeagueResult], min_n: int = 0, only_validated: bool = False) -> dict:
    per_league = {}
    all_preds, all_outcomes, all_base, all_blocks = [], [], [], []

    for code, result in results.items():
        n = result.metrics.get("n") or 0
        improved = bool(
            n
            and result.metrics.get("brier") is not None
            and result.metrics["brier"] < (result.baseline.get("brier") or 1e9)
        )
        interval = result.inference or {}
        per_league[code] = {
            "status": result.status,
            "n": n,
            "brier": result.metrics.get("brier"),
            "logloss": result.metrics.get("logloss"),
            "accuracy": result.metrics.get("accuracy"),
            "base_brier": result.baseline.get("brier"),
            "base_logloss": result.baseline.get("logloss"),
            "base_accuracy": result.baseline.get("accuracy"),
            "coverage": round(result.coverage, 4),
            "improved": improved,
            "p_value": interval.get("p_value"),
            "q_value": None,
            "validated": False,
            "inference": interval,
        }

    p_values = {
        code: item["p_value"]
        for code, item in per_league.items()
        if item["n"] >= min_n and item["p_value"] is not None
    }
    fdr = inference.benjamini_hochberg(p_values)
    for code, item in per_league.items():
        q_value = fdr.get(code, {}).get("q")
        item["q_value"] = q_value
        item["validated"] = bool(
            item["n"] >= min_n and item["improved"] and q_value is not None and q_value <= 0.05
        )

    total = 0
    weighted = {"brier": 0.0, "logloss": 0.0, "accuracy": 0.0}
    base = {"brier": 0.0, "logloss": 0.0, "accuracy": 0.0}
    bins = [{"count": 0, "pred": 0.0, "obs": 0.0} for _ in range(10)]

    for code, item in per_league.items():
        if only_validated and not item["validated"]:
            continue
        n = item["n"]
        if n:
            total += n
            for key in weighted:
                weighted[key] += item[key] * n
                base[key] += (item[f"base_{key}"] or 0.0) * n
            result = results[code]
            all_preds.extend(result.preds)
            all_outcomes.extend(result.outcomes)
            all_base.extend(result.base_preds)
            all_blocks.extend(f"{code}|{block}" for block in result.blocks)
        for bucket in results[code].reliability:
            if bucket["count"] and bucket["predicted"] is not None:
                index = min(int(float(bucket["bin"].split("-")[0]) * 10), 9)
                bins[index]["count"] += bucket["count"]
                bins[index]["pred"] += bucket["predicted"] * bucket["count"]
                bins[index]["obs"] += bucket["observed"] * bucket["count"]

    reliability = [
        {
            "bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}",
            "count": entry["count"],
            "predicted": round(entry["pred"] / entry["count"], 4) if entry["count"] else None,
            "observed": round(entry["obs"] / entry["count"], 4) if entry["count"] else None,
        }
        for i, entry in enumerate(bins)
    ]

    if total:
        weighted = {k: v / total for k, v in weighted.items()}
        base = {k: v / total for k, v in base.items()}

    global_inference = {}
    if all_preds:
        global_inference = inference.block_bootstrap(all_preds, all_outcomes, all_blocks, all_base)
        global_inference["brier_decomposition"] = inference.brier_decomposition(all_preds, all_outcomes)
        global_inference["ece"] = round(inference.expected_calibration_error(all_preds, all_outcomes), 5)

    return {
        "n": total,
        "brier": weighted["brier"] if total else None,
        "logloss": weighted["logloss"] if total else None,
        "accuracy": weighted["accuracy"] if total else None,
        "base_brier": base["brier"] if total else None,
        "base_logloss": base["logloss"] if total else None,
        "base_accuracy": base["accuracy"] if total else None,
        "reliability": reliability,
        "inference": global_inference,
        "leagues": per_league,
    }
