"""Genera los artefactos JSON que consume la web estática."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import backtest, ingest, ratings, stats
from .leagues import LEAGUES, SPORTS

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "site" / "data"

SPORTS_INDEX = [
    {"id": "football", "name": "Fútbol", "status": "active", "reason": "Resultados históricos de 40 competiciones (2010-2026)."},
    {"id": "basketball", "name": "Baloncesto", "status": "no-data", "reason": "Sin partidos ni marcadores en el repositorio."},
    {"id": "baseball", "name": "Béisbol", "status": "no-data", "reason": "Sin partidos ni marcadores en el repositorio."},
    {"id": "american-football", "name": "Fútbol americano", "status": "no-data", "reason": "Sin partidos ni marcadores en el repositorio."},
    {"id": "ice-hockey", "name": "Hockey hielo", "status": "no-data", "reason": "Sin partidos ni marcadores en el repositorio."},
    {"id": "tennis", "name": "Tenis", "status": "no-data", "reason": "Sin partidos ni marcadores en el repositorio."},
]


def _round(value: float | None, digits: int = 4):
    return None if value is None else round(float(value), digits)


def build(
    raw: Path = ingest.DEFAULT_RAW,
    cache: Path = ingest.DEFAULT_CACHE,
    out_dir: Path = OUT_DIR,
    aliases: Path | None = None,
    extra: Path | None = ingest.DEFAULT_EXTRA,
) -> dict:
    matches, rejected = ingest.ingest(raw, cache, aliases or ingest.DEFAULT_ALIASES, extra)
    if not matches:
        raise RuntimeError("No hay partidos normalizados: revisa la fuente de datos")

    groups: dict[str, list[dict]] = {}
    for match in matches:
        groups.setdefault(match["l"], []).append(match)

    results: dict[str, backtest.LeagueResult] = {}
    for code in sorted(groups, key=lambda c: LEAGUES.get(c, ("", "", 999))[2]):
        if code not in LEAGUES:
            continue
        rows = groups[code]
        print(f"  [{code}] {len(rows)} partidos...", flush=True)
        try:
            results[code] = backtest.evaluate_league(rows)
        except ValueError as error:
            print(f"  [{code}] omitida: {error}", flush=True)

    metrics = backtest.aggregate(results, min_n=200)
    metrics_validated = backtest.aggregate(results, min_n=200, only_validated=True)
    summary = ingest.summarize(matches)
    data_through = max(match["d"] for match in matches)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    leagues_json, ratings_json = [], {"sport": "football", "leagues": {}}
    for code, result in results.items():
        country, name, priority = LEAGUES[code]
        info = summary[code]
        model = result.model
        leagues_json.append(
            {
                "code": code,
                "country": country,
                "name": name,
                "priority": priority,
                "status": result.status,
                "seasons": info["seasons"],
                "matches": info["matches"],
                "teams": info["teams"],
                "first": info["first"],
                "last": info["last"],
                "coverage": _round(result.coverage),
                "validated": metrics["leagues"][code]["validated"],
                "tuning": {"half_life": model.half_life, "l2": model.l2, "alpha": model.extra.get("alpha", 0.0)},
            }
        )
        ratings_json["leagues"][code] = {
            "name": name,
            "country": country,
            "mu": _round(model.mu, 5),
            "home_adv": _round(model.home_adv, 5),
            "rho": _round(model.rho, 5),
            "n": model.n_matches,
            "data_through": model.data_through,
            "alpha": round(float(model.extra.get("alpha", 0.0)), 2),
            "base": [
                round(float(model.extra.get("base_probs", {}).get("home", 1 / 3)), 4),
                round(float(model.extra.get("base_probs", {}).get("draw", 1 / 3)), 4),
                round(float(model.extra.get("base_probs", {}).get("away", 1 / 3)), 4),
            ],
            "teams": [
                [team, _round(r.attack, 4), _round(r.defense, 4), r.matches, r.last]
                for team, r in sorted(model.teams.items())
            ],
        }

    ratings_json["generated_at"] = generated
    ratings_json["data_through"] = data_through
    ratings_json["home_goal_boost_pct"] = {
        code: round(100 * (2.718281828 ** result.model.home_adv - 1), 1)
        for code, result in results.items()
    }

    metrics_json = {
        "sport": "football",
        "generated_at": generated,
        "data_through": data_through,
        "totals": {"matches": len(matches), "leagues": len(results)},
        "rejected": dict(rejected),
        "aggregate": {k: v for k, v in metrics.items() if k != "leagues"},
        "aggregate_validated": {k: v for k, v in metrics_validated.items() if k != "leagues"},
        "min_evaluation_n": 200,
        "leagues": metrics["leagues"],
        "model": {
            "name": "Dixon-Coles con decaimiento temporal",
            "target": SPORTS["football"]["score_unit"],
            "hyperparameters": "elegidos por liga con holdout walk-forward",
        },
    }

    # --- Laboratorio estadistico (por liga, cargado bajo demanda en la web) ---
    lab_index, lab_files = stats.build_lab(matches)
    advanced = stats.advanced_sample(extra)
    overall = stats.totals(matches)
    lab_meta = {
        "generated_at": generated,
        "data_through": data_through,
        "sport": "football",
        "sports": SPORTS_INDEX,
        "totals": {
            "matches": overall["matches"],
            "leagues": len(results),
            "teams": sum(item["teams"] for item in lab_index.values()),
            **{k: v for k, v in overall.items() if k != "matches"},
        },
        "data_quality": {"rejected": dict(rejected), "advanced_sample": advanced},
        "leagues": [],
    }
    for code in sorted(lab_index, key=lambda c: LEAGUES[c][2]):
        country, name, priority = LEAGUES[code]
        model_meta = metrics["leagues"].get(code, {})
        lab_meta["leagues"].append(
            {
                "code": code,
                "country": country,
                "name": name,
                "priority": priority,
                "status": model_meta.get("status"),
                "validated": model_meta.get("validated"),
                **lab_index[code],
            }
        )

    lab_dir = out_dir / "lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "index.json").write_text(
        json.dumps(lab_meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    for code, payload in lab_files.items():
        country, name, _ = LEAGUES[code]
        body = {
            "code": code,
            "country": country,
            "name": name,
            "data_through": max(row["d"] for row in matches if row["l"] == code),
            **payload,
        }
        (lab_dir / f"{code}.json").write_text(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
    print(f"  escrito {out_dir.name}/lab/ ({len(lab_files)} ligas)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "leagues.json": sorted(leagues_json, key=lambda item: item["priority"]),
        "ratings.json": ratings_json,
        "metrics.json": metrics_json,
    }
    for filename, payload in files.items():
        path = out_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"  escrito {out_dir.name}/{filename} ({path.stat().st_size / 1024:.0f} KB)", flush=True)

    return {
        "leagues": len(leagues_json),
        "matches": len(matches),
        "aggregate": metrics,
        "rejected": dict(rejected),
    }
