"""Estadística descriptiva del laboratorio.

Todo se calcula sobre los partidos normalizados. No se inventa ninguna señal:
si un dato no existe (xG, jugadores, minutos de gol), no aparece.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import ingest, ratings

PYTH_EXP = 1.7
FORM_LENGTH = 10
TOP_SCORELINES = 12


def _result(match: dict) -> int:
    return 0 if match["hg"] > match["ag"] else (1 if match["hg"] == match["ag"] else 2)


def _empty_team(name: str) -> dict:
    split = {"m": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0}
    return {
        "name": name,
        "m": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0,
        "home": dict(split), "away": dict(split),
        "over25": 0, "btts": 0, "cs": 0, "fts": 0,
        "ht_m": 0, "ht_lead": 0, "ht_trail": 0, "comeback": 0, "collapse": 0,
        "gf1h": 0, "ga1h": 0, "gf2h": 0, "ga2h": 0,
        "pyth": 0.0, "first": "", "last": "", "_form": [],
    }


def _pyth_points(gf: int, ga: int) -> float:
    num = gf**PYTH_EXP
    den = num + ga**PYTH_EXP
    return 3.0 * (num / den) if den else 1.5


def _add_team(table: dict, name: str, gf: int, ga: int, venue: str, ht: list[int] | None) -> None:
    team = table.setdefault(name, _empty_team(name))
    win, draw = gf > ga, gf == ga
    pts = 3 if win else (1 if draw else 0)
    team["m"] += 1
    team["w"] += win
    team["d"] += draw
    team["l"] += not (win or draw)
    team["gf"] += gf
    team["ga"] += ga
    team["pts"] += pts
    side = team[venue]
    side["m"] += 1
    side["w"] += win
    side["d"] += draw
    side["l"] += not (win or draw)
    side["gf"] += gf
    side["ga"] += ga
    side["pts"] += pts
    team["over25"] += gf + ga > 2
    team["btts"] += gf > 0 and ga > 0
    team["cs"] += ga == 0
    team["fts"] += gf == 0
    team["pyth"] += _pyth_points(gf, ga)
    if ht:
        ht_for, ht_ag = ht
        team["ht_m"] += 1
        team["gf1h"] += ht_for
        team["ga1h"] += ht_ag
        team["gf2h"] += gf - ht_for
        team["ga2h"] += ga - ht_ag
        if ht_for > ht_ag:
            team["ht_lead"] += 1
            if not win:
                team["collapse"] += 1
        elif ht_for < ht_ag:
            team["ht_trail"] += 1
            if win:
                team["comeback"] += 1
    team["_form"].append("W" if win else ("D" if draw else "L"))


def _rate(value: int, total: int, digits: int = 4) -> float:
    return round(value / total, digits) if total else 0.0


def totals(matches: list[dict]) -> dict:
    n = len(matches)
    goals = home = away = over15 = over25 = over35 = btts = nil = clean = 0
    home_w = draws = 0
    ht_n = ht_draw = ht_o05 = ht_o15 = g1h = g2h = comebacks = leads_lost = 0
    for match in matches:
        hg, ag = match["hg"], match["ag"]
        goals += hg + ag
        home += hg
        away += ag
        over15 += hg + ag > 1
        over25 += hg + ag > 2
        over35 += hg + ag > 3
        btts += hg > 0 and ag > 0
        nil += hg == 0 and ag == 0
        clean += hg == 0 or ag == 0
        home_w += hg > ag
        draws += hg == ag
        ht = match.get("ht")
        if ht:
            ht_n += 1
            hth, hta = ht
            ht_draw += hth == hta
            ht_o05 += hth + hta > 0
            ht_o15 += hth + hta > 1
            g1h += hth + hta
            g2h += (hg + ag) - (hth + hta)
            if hth > hta and hg < ag:
                comebacks += 1
                leads_lost += 1
            elif hta > hth and hg > ag:
                comebacks += 1
                leads_lost += 1
    return {
        "matches": n,
        "goals": goals,
        "goals_home": home,
        "goals_away": away,
        "avg_goals": round(goals / n, 3) if n else 0.0,
        "avg_home": round(home / n, 3) if n else 0.0,
        "avg_away": round(away / n, 3) if n else 0.0,
        "home_win": _rate(home_w, n),
        "draw": _rate(draws, n),
        "away_win": _rate(n - home_w - draws, n),
        "over15": _rate(over15, n),
        "over25": _rate(over25, n),
        "over35": _rate(over35, n),
        "btts": _rate(btts, n),
        "clean_sheet": _rate(clean, n),
        "nil_nil": _rate(nil, n),
        "ht_matches": ht_n,
        "ht_coverage": _rate(ht_n, n),
        "ht_draw": _rate(ht_draw, ht_n),
        "ht_over05": _rate(ht_o05, ht_n),
        "ht_over15": _rate(ht_o15, ht_n),
        "avg_goals_1h": round(g1h / ht_n, 3) if ht_n else 0.0,
        "avg_goals_2h": round(g2h / ht_n, 3) if ht_n else 0.0,
        "comebacks": comebacks,
        "comeback_rate": _rate(comebacks, ht_n),
        "leads_lost": leads_lost,
        "lead_lost_rate": _rate(leads_lost, ht_n),
    }


def league_lab(matches: list[dict]) -> dict:
    matches = sorted(matches, key=lambda m: m["d"])
    table: dict[str, dict] = {}
    scorelines = Counter()
    distribution = Counter()
    referees: dict[str, dict] = {}
    seasons: dict[str, list[dict]] = {}

    for match in matches:
        hg, ag = match["hg"], match["ag"]
        _add_team(table, match["h"], hg, ag, "home", match.get("ht"))
        ht_away = list(reversed(match["ht"])) if match.get("ht") else None
        _add_team(table, match["a"], ag, hg, "away", ht_away)
        scorelines[(hg, ag)] += 1
        distribution[min(hg + ag, 8)] += 1
        seasons.setdefault(match["s"], []).append(match)
        for side in ("h", "a"):
            team = table[match[side]]
            if not team["first"] or match["d"] < team["first"]:
                team["first"] = match["d"]
            if match["d"] > team["last"]:
                team["last"] = match["d"]
        referee = match.get("ref")
        if referee:
            entry = referees.setdefault(
                referee, {"name": referee, "matches": 0, "goals": 0, "home": 0, "draw": 0, "over25": 0}
            )
            entry["matches"] += 1
            entry["goals"] += hg + ag
            entry["home"] += hg > ag
            entry["draw"] += hg == ag
            entry["over25"] += hg + ag > 2

    elo = ratings.elo_ratings(matches)
    names = sorted(table)
    team_rows = []
    for name in names:
        team = table[name]
        form = "".join(reversed(team.pop("_form")[-FORM_LENGTH:]))
        streak = 0
        if form:
            streak = len(form) - len(form.lstrip(form[0]))
        pyth = team["pyth"]
        team.update(
            {
                "ppg": round(team["pts"] / team["m"], 3) if team["m"] else 0.0,
                "gd": team["gf"] - team["ga"],
                "over25_rate": _rate(team["over25"], team["m"]),
                "btts_rate": _rate(team["btts"], team["m"]),
                "cs_rate": _rate(team["cs"], team["m"]),
                "fts_rate": _rate(team["fts"], team["m"]),
                "comeback_rate": _rate(team["comeback"], team["ht_m"]),
                "collapse_rate": _rate(team["collapse"], team["ht_m"]),
                "form": form,
                "streak": f"{form[0]}{streak}" if form else "",
                "elo": round(elo.get(name, 1500.0), 1),
                "pyth": round(pyth, 1),
                "luck": round(team["pts"] - pyth, 1),
            }
        )
        team_rows.append(team)

    ref_rows = []
    for entry in referees.values():
        if entry["matches"] < 8:
            continue
        ref_rows.append(
            {
                **entry,
                "avg_goals": round(entry["goals"] / entry["matches"], 3),
                "home_win": _rate(entry["home"], entry["matches"]),
                "draw": _rate(entry["draw"], entry["matches"]),
                "over25": _rate(entry["over25"], entry["matches"]),
            }
        )
    ref_rows.sort(key=lambda row: -row["matches"])

    season_rows = [
        {"season": season, **totals(rows)}
        for season, rows in sorted(seasons.items(), key=lambda item: min(m["d"] for m in item[1]))
    ]

    index = {name: i for i, name in enumerate(names)}
    compact = [
        [
            match["d"],
            index[match["h"]],
            index[match["a"]],
            match["hg"],
            match["ag"],
            (match["ht"][0] if match.get("ht") else -1),
            (match["ht"][1] if match.get("ht") else -1),
        ]
        for match in matches
    ]

    return {
        "totals": totals(matches),
        "seasons": season_rows,
        "goals_distribution": [distribution.get(goals, 0) for goals in range(8)] + [sum(v for k, v in distribution.items() if k >= 8)],
        "scorelines": [[hg, ag, count] for (hg, ag), count in scorelines.most_common(TOP_SCORELINES)],
        "teamNames": names,
        "teams": team_rows,
        "matches": compact,
        "referees": ref_rows,
    }


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADVANCED_CACHE = ROOT / "data" / "advanced_sample.json"


def advanced_sample(
    extra_dir: Path | None = ingest.DEFAULT_EXTRA,
    cache: Path = DEFAULT_ADVANCED_CACHE,
) -> dict:
    """Muestra piloto de estadísticas de API-Football (tiros, posesión, pases)."""
    if extra_dir is None or not Path(extra_dir).exists():
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        return {"n": 0, "rows": [], "note": "sin cache local"}
    result = _advanced_from_raw(Path(extra_dir))
    if result["n"]:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return result


def _advanced_from_raw(root: Path) -> dict:
    files = sorted(root.glob("api_football/**/stats_*.json"))
    fixtures: dict[int, dict] = {}
    for path in sorted((root / "api_football").glob("*/fixtures.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for item in payload.get("response", []):
            fixtures[item.get("fixture", {}).get("id")] = item

    seen: set[int] = set()
    rows = []
    pairs: list[tuple[float, float]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        response = payload.get("response") or []
        if not response:
            continue
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        fixture_id = int(digits) if digits else None
        if fixture_id in seen:
            continue
        seen.add(fixture_id)
        stats = {}
        for side in response[:2]:
            name = side.get("team", {}).get("name")
            values = {item["type"]: item["value"] for item in side.get("statistics", [])}
            stats[name] = values
        names = list(stats)
        if len(names) != 2:
            continue
        fixture = fixtures.get(fixture_id)
        if fixture:
            home = fixture["teams"]["home"]["name"]
            away = fixture["teams"]["away"]["name"]
            score = fixture.get("goals") or {}
            hg, ag = score.get("home"), score.get("away")
            day = fixture["fixture"]["date"][:10]
        else:
            home, away, hg, ag, day = names[0], names[1], None, None, ""

        def number(name: str, key: str):
            value = stats.get(name, {}).get(key)
            return value if isinstance(value, (int, float)) else None

        def ratio(name: str, key: str):
            value = stats.get(name, {}).get(key)
            if isinstance(value, str) and value.endswith("%"):
                return float(value[:-1])
            return None

        rows.append(
            {
                "date": day,
                "home": home,
                "away": away,
                "hg": hg,
                "ag": ag,
                "shots_home": number(home, "Total Shots"),
                "shots_away": number(away, "Total Shots"),
                "sot_home": number(home, "Shots on Goal"),
                "sot_away": number(away, "Shots on Goal"),
                "pos_home": ratio(home, "Ball Possession"),
                "pos_away": ratio(away, "Ball Possession"),
                "pass_home": ratio(home, "Passes %"),
                "pass_away": ratio(away, "Passes %"),
                "corners_home": number(home, "Corner Kicks"),
                "corners_away": number(away, "Corner Kicks"),
                "fouls_home": number(home, "Fouls"),
                "fouls_away": number(away, "Fouls"),
            }
        )
        if isinstance(hg, int) and isinstance(ag, int):
            sot_h, sot_a = number(home, "Shots on Goal"), number(away, "Shots on Goal")
            if sot_h is not None and sot_a is not None:
                pairs.append((float(sot_h - sot_a), float(hg - ag)))

    correlation = None
    if len(pairs) >= 5:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        correlation = round(cov / (vx * vy), 3) if vx and vy else None

    return {
        "n": len(rows),
        "rows": rows,
        "correlation_sot_goals": correlation,
        "note": f"Muestra piloto de API-Football: {len(rows)} partidos de Colombia 2022-2024. Sin xG (el campo llega nulo).",
    }


def build_lab(matches: list[dict]) -> tuple[dict, dict[str, dict]]:
    groups: dict[str, list[dict]] = {}
    for match in matches:
        groups.setdefault(match["l"], []).append(match)
    index = {}
    files = {}
    for code, rows in sorted(groups.items()):
        lab = league_lab(rows)
        last_season = max(rows, key=lambda m: m["d"])["s"]
        recent = league_lab([m for m in rows if m["s"] == last_season])
        index[code] = {
            "matches": lab["totals"]["matches"],
            "goals": lab["totals"]["goals"],
            "avg_goals": lab["totals"]["avg_goals"],
            "home_win": lab["totals"]["home_win"],
            "draw": lab["totals"]["draw"],
            "away_win": lab["totals"]["away_win"],
            "over25": lab["totals"]["over25"],
            "btts": lab["totals"]["btts"],
            "ht_coverage": lab["totals"]["ht_coverage"],
            "comeback_rate": lab["totals"]["comeback_rate"],
            "teams": len(lab["teamNames"]),
            "seasons": len(lab["seasons"]),
            "first": min(row["d"] for row in rows),
            "last": max(row["d"] for row in rows),
        }
        files[code] = {
            **lab,
            "recent": {"season": last_season, "totals": recent["totals"], "teams": recent["teams"]},
        }
    return index, files
