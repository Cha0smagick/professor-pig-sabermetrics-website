"""Benchmark y modelo híbrido Dixon-Coles + XGBoost.

Reglas para no mentir:
- Las features de cada partido se calculan solo con partidos anteriores.
- El XGBoost de evaluacion se entrena con un corte temporal y se prueba despues.
- El peso del ensemble (50/50) se fija a priori, no se optimiza sobre el test.
- Si XGBoost no esta instalado, el sistema sigue funcionando solo con DC.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import date, timedelta

import numpy as np

from . import inference, ratings
from .backtest import LeagueResult, _reliability
from .model import outcome_probs

CUTOFF = "2024-07-01"
WEIGHT_XGB = 0.5
FEATURES = [
    "elo_local", "elo_visitante", "elo_diff", "elo_esperado_local",
    "forma_local", "forma_visitante", "gf5_local", "ga5_local",
    "gf5_visitante", "ga5_visitante", "descanso_local", "descanso_visitante",
    "partidos_local", "partidos_visitante",
]


def _empty() -> dict:
    return {
        "elo": 1500.0,
        "pts": deque(maxlen=5),
        "gf": deque(maxlen=5),
        "ga": deque(maxlen=5),
        "last": None,
        "played": 0,
    }


def _row_features(state: dict, home: str, away: str, day: date) -> list[float]:
    sh = state.get(home) or _empty()
    sa = state.get(away) or _empty()
    eh, ea = sh["elo"], sa["elo"]
    form_h = sum(sh["pts"]) / 15 if sh["played"] else 0.5
    form_a = sum(sa["pts"]) / 15 if sa["played"] else 0.5
    gf_h = sum(sh["gf"]) / max(len(sh["gf"]), 1)
    ga_h = sum(sh["ga"]) / max(len(sh["ga"]), 1)
    gf_a = sum(sa["gf"]) / max(len(sa["gf"]), 1)
    ga_a = sum(sa["ga"]) / max(len(sa["ga"]), 1)
    rest_h = min(max((day - sh["last"]).days, 0), 60) if sh["last"] else 60
    rest_a = min(max((day - sa["last"]).days, 0), 60) if sa["last"] else 60
    expected = 1.0 / (1.0 + 10 ** ((ea - (eh + 60)) / 400.0))
    return [
        eh, ea, eh - ea, expected,
        form_h, form_a, gf_h, ga_h, gf_a, ga_a,
        rest_h, rest_a, np.log1p(sh["played"]), np.log1p(sa["played"]),
    ]


def _update(state: dict, match: dict) -> None:
    day = date.fromisoformat(match["d"])
    sh = state.setdefault(match["h"], _empty())
    sa = state.setdefault(match["a"], _empty())
    expected = 1.0 / (1.0 + 10 ** ((sa["elo"] - (sh["elo"] + 60)) / 400.0))
    score = 1.0 if match["hg"] > match["ag"] else (0.5 if match["hg"] == match["ag"] else 0.0)
    delta = 20 * (score - expected)
    sh["elo"] += delta
    sa["elo"] -= delta
    hp = 3 if match["hg"] > match["ag"] else (1 if match["hg"] == match["ag"] else 0)
    ap = 3 if match["ag"] > match["hg"] else (1 if match["hg"] == match["ag"] else 0)
    sh["pts"].append(hp)
    sa["pts"].append(ap)
    sh["gf"].append(match["hg"]); sh["ga"].append(match["ag"])
    sa["gf"].append(match["ag"]); sa["ga"].append(match["hg"])
    sh["last"] = day; sa["last"] = day
    sh["played"] += 1; sa["played"] += 1


def build_features(matches: list[dict]) -> tuple[np.ndarray, np.ndarray, list[tuple], dict]:
    state: dict = {}
    rows, outcomes, keys = [], [], []
    for match in sorted(matches, key=lambda m: m["d"]):
        day = date.fromisoformat(match["d"])
        rows.append(_row_features(state, match["h"], match["a"], day))
        outcomes.append(0 if match["hg"] > match["ag"] else (1 if match["hg"] == match["ag"] else 2))
        keys.append((match["d"], match["h"], match["a"]))
        _update(state, match)
    return np.asarray(rows, dtype=float), np.asarray(outcomes, dtype=int), keys, state


def _design(numeric: np.ndarray, leagues: list[str]) -> tuple[np.ndarray, list[str]]:
    codes = sorted(set(leagues))
    index = {code: i for i, code in enumerate(codes)}
    onehot = np.zeros((len(leagues), len(codes)), dtype=float)
    for row, code in enumerate(leagues):
        onehot[row, index[code]] = 1.0
    return np.hstack([numeric, onehot]), codes


def _xgb():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def _mix(dc: dict, xgb_probs: np.ndarray) -> dict:
    p = WEIGHT_XGB
    return {
        "home": (1 - p) * dc["home"] + p * float(xgb_probs[0]),
        "draw": (1 - p) * dc["draw"] + p * float(xgb_probs[1]),
        "away": (1 - p) * dc["away"] + p * float(xgb_probs[2]),
    }


def _block_key(key: tuple) -> str:
    day = date.fromisoformat(key[0])
    return f"{key[0][:4]}W{day.isocalendar().week:02d}"


def evaluate_hybrid(matches: list[dict], results: dict[str, LeagueResult]) -> tuple[dict, dict, dict | None]:
    """Recalcula cada liga con el ensemble y devuelve benchmark + artefacto web."""
    model = _xgb()
    all_keys = [key for result in results.values() for key in result.keys]
    if model is None or not all_keys:
        return results, {"available": False, "note": "XGBoost no instalado o sin partidos de test"}, None

    numeric, outcomes, keys, _ = build_features(matches)
    leagues = [m["l"] for m in sorted(matches, key=lambda m: m["d"])]
    design, codes = _design(numeric, leagues)
    dates = np.array([key[0] for key in keys])
    index_by_key = {key: i for i, key in enumerate(keys)}

    season_by_key: dict[tuple, str] = {}
    for result in results.values():
        for key, block in zip(result.keys, result.blocks):
            season_by_key[key] = block.rsplit("-W", 1)[0] if "-W" in block else block

    # Un XGBoost por temporada de test, entrenado solo con el pasado.
    by_key: dict[tuple, np.ndarray] = {}
    for season in sorted(set(season_by_key.values())):
        season_keys = [key for key, value in season_by_key.items() if value == season]
        start = min(key[0] for key in season_keys)
        train_mask = dates < start
        if train_mask.sum() < 5000:
            continue
        model.fit(design[train_mask], outcomes[train_mask])
        rows = [index_by_key[key] for key in season_keys]
        proba = model.predict_proba(design[rows])
        for key, probability in zip(season_keys, proba):
            by_key[key] = probability

    hybrid_results: dict[str, LeagueResult] = {}
    for code, result in results.items():
        preds, base_preds, dc_preds, outs, blocks = [], [], [], [], []
        for key, dc, base in zip(result.keys, result.preds, result.base_preds):
            if key not in by_key:
                continue
            preds.append(_mix(dc, by_key[key]))
            dc_preds.append(dc)
            base_preds.append(base)
            outs.append(outcomes[index_by_key[key]])
            blocks.append(f"{code}|{_block_key(key)}")
        if not preds:
            continue
        metrics = inference.metric_set(
            np.array([[p["home"], p["draw"], p["away"]] for p in preds]), np.array(outs)
        )
        baseline = inference.metric_set(
            np.array([[p["home"], p["draw"], p["away"]] for p in base_preds]), np.array(outs)
        )
        interval = inference.block_bootstrap(preds, outs, blocks, base_preds)
        interval["vs_dc"] = inference.block_bootstrap(preds, outs, blocks, dc_preds)
        interval["brier_decomposition"] = inference.brier_decomposition(preds, outs)
        interval["ece"] = round(inference.expected_calibration_error(preds, outs), 5)
        hybrid_results[code] = LeagueResult(
            model=result.model,
            status=result.status,
            metrics=metrics,
            baseline=baseline,
            reliability=_reliability(preds, outs),
            tuning=result.tuning,
            coverage=result.coverage,
            preds=preds,
            base_preds=base_preds,
            outcomes=outs,
            blocks=blocks,
            keys=[key for key in result.keys if key in by_key],
            inference=interval,
        )

    benchmark = _fixed_benchmark(matches, numeric, outcomes, keys, design, codes, model, results)
    artifact = _deployment_artifact(matches, numeric, leagues, codes)
    return hybrid_results, benchmark, artifact


def _fixed_benchmark(matches, numeric, outcomes, keys, design, codes, model, results) -> dict:
    """Comparacion a corte fijo: baseline, DC, XGBoost e hibrido sobre los mismos partidos."""
    dates = np.array([key[0] for key in keys])
    train = dates < CUTOFF
    test = ~train
    if train.sum() < 5000 or test.sum() < 500:
        return {"available": False, "note": "corte sin muestra suficiente"}

    model.fit(design[train], outcomes[train])
    xgb = model.predict_proba(design[test])
    test_keys = [key for key, keep in zip(keys, test) if keep]
    by_key = {key: xgb[i] for i, key in enumerate(test_keys)}
    index_by_key = {key: i for i, key in enumerate(keys)}
    league_by_key = {(m["d"], m["h"], m["a"]): m["l"] for m in matches}

    groups: dict[str, list[dict]] = defaultdict(list)
    for match in matches:
        groups[match["l"]].append(match)
    dc, base = {}, {}
    for code, rows in groups.items():
        tr = [m for m in rows if m["d"] < CUTOFF]
        if len(tr) < 50:
            continue
        counts = Counter(
            0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2) for m in tr
        )
        n = sum(counts.values()) or 1
        base[code] = {"home": counts[0] / n, "draw": counts[1] / n, "away": counts[2] / n}
        production = results.get(code)
        half_life = production.model.half_life if production else ratings.DEFAULT_HALF_LIFE
        l2 = production.model.l2 if production else ratings.DEFAULT_L2
        alpha = production.model.extra.get("alpha", 0.0) if production else 0.0
        try:
            dc_model = ratings.fit(tr, half_life=half_life, l2=l2)
        except ValueError:
            continue
        for m in rows:
            if m["d"] < CUTOFF:
                continue
            key = (m["d"], m["h"], m["a"])
            if key in by_key and m["h"] in dc_model.teams and m["a"] in dc_model.teams:
                probs = outcome_probs(ratings.score_matrix(dc_model, m["h"], m["a"]))
                if alpha:
                    probs = {k: (1 - alpha) * probs[k] + alpha * base[code][k] for k in probs}
                dc[key] = probs

    dc_preds, xgb_preds, base_preds, outs, blocks = [], [], [], [], []
    for key in test_keys:
        code = league_by_key.get(key)
        if key not in dc or code not in base:
            continue
        dc_preds.append(dc[key])
        xgb_preds.append(dict(zip(("home", "draw", "away"), by_key[key])))
        base_preds.append(base[code])
        outs.append(outcomes[index_by_key[key]])
        blocks.append(f"{code}|{_block_key(key)}")
    if not outs:
        return {"available": False, "note": "sin interseccion DC/XGB"}

    hybrid = [{k: (1 - WEIGHT_XGB) * d[k] + WEIGHT_XGB * x[k] for k in ("home", "draw", "away")}
              for d, x in zip(dc_preds, xgb_preds)]

    def evaluate(name, preds, base_for=None):
        arr = np.array([[p["home"], p["draw"], p["away"]] for p in preds])
        metrics = inference.metric_set(arr, np.array(outs))
        interval = inference.block_bootstrap(preds, outs, blocks, base_for) if base_for is not None else {}
        return {
            "n": metrics["n"],
            "brier": round(metrics["brier"], 5),
            "logloss": round(metrics["logloss"], 5),
            "accuracy": round(metrics["accuracy"], 5),
            "inference": interval,
            "brier_decomposition": inference.brier_decomposition(preds, outs),
            "ece": round(inference.expected_calibration_error(preds, outs), 5),
        }

    return {
        "available": True,
        "cutoff": CUTOFF,
        "weights": {"dc": 1 - WEIGHT_XGB, "xgboost": WEIGHT_XGB},
        "n": len(outs),
        "models": {
            "baseline": evaluate("baseline", base_preds),
            "dc": evaluate("dc", dc_preds, base_preds),
            "xgboost": evaluate("xgboost", xgb_preds, base_preds),
            "hybrid": evaluate("hybrid", hybrid, base_preds),
            "hybrid_vs_dc": evaluate("hybrid_vs_dc", hybrid, dc_preds),
            "xgboost_vs_dc": evaluate("xgboost_vs_dc", xgb_preds, dc_preds),
            "hybrid_vs_xgboost": evaluate("hybrid_vs_xgboost", hybrid, xgb_preds),
        },
    }


def _deployment_artifact(matches, numeric, leagues, codes) -> dict:
    """Modelo final (todo el historico) y probabilidades para todos los pares."""
    model = _xgb()
    if model is None:
        return {}
    model.fit(_design(numeric, leagues)[0], np.array(
        [0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)
         for m in sorted(matches, key=lambda m: m["d"])]
    ))
    _, _, _, state = build_features(matches)
    last_day = max(date.fromisoformat(m["d"]) for m in matches)
    snapshot = last_day + timedelta(days=3)

    groups: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        groups[match["l"]].update((match["h"], match["a"]))

    artifact = {
        "generated_at": snapshot.isoformat(),
        "weights": {"dc": 1 - WEIGHT_XGB, "xgboost": WEIGHT_XGB},
        "features": FEATURES,
        "leagues": {},
    }
    code_index = {code: i for i, code in enumerate(codes)}
    for code, teams in sorted(groups.items()):
        names = sorted(teams)
        rows = []
        for home in names:
            for away in names:
                rows.append(_row_features(state, home, away, snapshot))
        design = np.zeros((len(rows), len(FEATURES) + len(codes)), dtype=float)
        design[:, : len(FEATURES)] = np.asarray(rows)
        design[:, len(FEATURES) + code_index[code]] = 1.0
        probs = model.predict_proba(design)
        flat = []
        for prob in probs:
            flat.extend([int(round(1000 * value)) for value in prob])
        assert len(flat) == len(names) * len(names) * 3
        artifact["leagues"][code] = {"n": len(names), "teams": names, "pairs": flat}
    return artifact
