"""Actualización diaria de resultados desde fuentes gratuitas.

Diseño:
- Fútbol: openfootball (raw de la temporada en curso) + OpenLigaDB + TheSportsDB.
- Otros deportes: MLB StatsAPI, NHL API, nflverse y TheSportsDB.
- Cada fallo de red se registra y no rompe el proceso.
- Los resultados de fútbol se añaden a `data/live_matches.jsonl` y entran al
  modelo en el siguiente build. El resto se guarda como eventos en
  `data/live_events.jsonl` (aún sin modelo predictivo).
"""

from __future__ import annotations

import csv
import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .leagues import LEAGUES

ROOT = Path(__file__).resolve().parent.parent
LIVE_MATCHES = ROOT / "data" / "live_matches.jsonl"
LIVE_EVENTS = ROOT / "data" / "live_events.jsonl"
CACHE_NFL = ROOT / "data" / "cache" / "nfl_games.csv"

OPENFOOTBALL = "https://raw.githubusercontent.com/openfootball/football.json/master"
OPENLIGADB = "https://api.openligadb.de/getmatchdata/bl1"
SPORTSDB = "https://www.thesportsdb.com/api/v1/json/123/eventsday.php"
MLB = "https://statsapi.mlb.com/api/v1/schedule"
NHL = "https://api-web.nhle.com/v1/score"
NFLVERSE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

CALENDAR_LEAGUES = {"mls", "br.1", "br.2", "ar.1", "co.1", "jp.1", "cn.1", "au.1"}
SPORTSDB_SPORTS = {
    "Soccer": "fútbol",
    "Basketball": "baloncesto",
    "Ice Hockey": "hockey hielo",
    "American Football": "fútbol americano",
    "Baseball": "béisbol",
    "Tennis": "tenis",
}
SPORTSDB_LEAGUES = {
    "English Premier League": "en.1",
    "Spanish La Liga": "es.1",
    "Italian Serie A": "it.1",
    "German Bundesliga": "de.1",
    "French Ligue 1": "fr.1",
    "American Major League Soccer": "mls",
    "Brazilian Serie A": "br.1",
    "Argentinian Primera Division": "ar.1",
    "Mexican Primera League": "mx.1",
    "Colombian Primera A": "co.1",
    "Dutch Eredivisie": "nl.1",
    "Portuguese Primeira Liga": "pt.1",
}


def _get(url: str, timeout: int = 20) -> str | None:
    request = Request(url, headers={"User-Agent": "Sabermetria/1.0 (personal research)"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None


def _json(url: str, timeout: int = 20):
    text = _get(url, timeout)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _ft(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _season_for(code: str, day: date) -> str:
    if code in CALENDAR_LEAGUES:
        return str(day.year)
    start = day.year if day.month >= 7 else day.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _football_row(code: str, season: str, day: date, home: str, away: str, hg: int, ag: int, ht, source: str) -> dict:
    row = {"l": code, "s": season, "d": day.isoformat(), "h": home, "a": away, "hg": hg, "ag": ag, "src": source}
    if isinstance(ht, (list, tuple)) and len(ht) == 2:
        try:
            row["ht"] = [int(ht[0]), int(ht[1])]
        except (TypeError, ValueError):
            pass
    elif hg == 0 and ag == 0:
        row["ht"] = [0, 0]
        row["hti"] = True
    return row


def _event(sport: str, league: str, day: date, home: str, away: str, hg, ag, source: str) -> dict:
    return {
        "sport": sport,
        "league": league,
        "date": day.isoformat(),
        "home": home,
        "away": away,
        "hg": _ft(hg),
        "ag": _ft(ag),
        "source": source,
    }


# ---------------- fútbol ----------------

def _openfootball_urls(day: date) -> list[tuple[str, str]]:
    year = day.year
    folders = [f"{year - 1}-{str(year)[-2:]}", f"{year}-{str(year + 1)[-2:]}", str(year)]
    return [(code, f"{OPENFOOTBALL}/{folder}/{code}.json") for code in LEAGUES for folder in folders]


def fetch_openfootball(days: set[date]) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    urls = []
    for day in days:
        urls.extend(_openfootball_urls(day))
    seen_urls = {}
    for code, url in urls:
        seen_urls[url] = code

    def load(item):
        url, code = item
        payload = _json(url, timeout=25)
        if payload is None:
            return code, None
        matches = []
        for match in payload.get("matches", []):
            try:
                match_day = date.fromisoformat(match.get("date", ""))
            except ValueError:
                continue
            if match_day not in days:
                continue
            score = (match.get("score") or {}).get("ft")
            if not (isinstance(score, list) and len(score) == 2):
                continue
            hg, ag = _ft(score[0]), _ft(score[1])
            home, away = match.get("team1"), match.get("team2")
            if hg is None or ag is None or not home or not away:
                continue
            season = url.rsplit("/", 2)[1]
            matches.append(_football_row(code, season, match_day, home, away, hg, ag, (match.get("score") or {}).get("ht"), "openfootball"))
        return code, matches

    with ThreadPoolExecutor(max_workers=8) as pool:
        for code, matches in pool.map(load, seen_urls.items()):
            if matches:
                rows.extend(matches)
    return rows, []


def fetch_openligadb(days: set[date]) -> tuple[list[dict], list[str]]:
    year = max(day.year for day in days)
    payload = _json(f"{OPENLIGADB}/{year}")
    if not isinstance(payload, list):
        return [], []
    rows = []
    for match in payload:
        try:
            day = datetime.fromisoformat(match.get("matchDateTime", "")).date()
        except ValueError:
            continue
        if day not in days or not match.get("matchIsFinished"):
            continue
        home = (match.get("team1") or {}).get("teamName")
        away = (match.get("team2") or {}).get("teamName")
        final = next((r for r in match.get("matchResults", []) if r.get("resultTypeID") == 2), None)
        half = next((r for r in match.get("matchResults", []) if r.get("resultTypeID") == 1), None)
        if not home or not away or final is None:
            continue
        hg, ag = _ft(final.get("pointsTeam1")), _ft(final.get("pointsTeam2"))
        if hg is None or ag is None:
            continue
        ht = [half.get("pointsTeam1"), half.get("pointsTeam2")] if half else None
        rows.append(_football_row("de.1", _season_for("de.1", day), day, home, away, hg, ag, ht, "openligadb"))
    return rows, []


def fetch_sportsdb(days: set[date]) -> tuple[list[dict], list[dict], list[str]]:
    matches, events, errors = [], [], []
    jobs = [(day, sport, label) for day in days for sport, label in SPORTSDB_SPORTS.items()]
    for day, sport, label in jobs:
        payload = None
        for attempt in range(3):
            payload = _json(f"{SPORTSDB}?d={day.isoformat()}&s={quote(sport)}")
            if payload is not None:
                break
            time.sleep(3 * (attempt + 1))
        if not isinstance(payload, dict):
            errors.append(f"thesportsdb:{sport}")
            continue
        for item in payload.get("events") or []:
            home, away = item.get("strHomeTeam"), item.get("strAwayTeam")
            hg, ag = _ft(item.get("intHomeScore")), _ft(item.get("intAwayScore"))
            if not home or not away:
                continue
            events.append(_event(label, item.get("strLeague") or sport, day, home, away, hg, ag, "thesportsdb"))
            if sport != "Soccer" or hg is None or ag is None:
                continue
            code = SPORTSDB_LEAGUES.get(item.get("strLeague"))
            if not code:
                continue
            matches.append(_football_row(code, _season_for(code, day), day, home, away, hg, ag, None, "thesportsdb"))
        time.sleep(2.2)
    return matches, events, errors


# ---------------- otros deportes ----------------

def fetch_mlb(days: set[date]) -> tuple[list[dict], list[str]]:
    events, errors = [], []
    for day in days:
        payload = _json(f"{MLB}?sportId=1&date={day.isoformat()}")
        if not isinstance(payload, dict):
            errors.append(f"mlb:{day}")
            continue
        for block in payload.get("dates", []):
            for game in block.get("games", []):
                if (game.get("status") or {}).get("abstractGameState") != "Final":
                    continue
                teams = game.get("teams", {})
                home = (teams.get("home") or {}).get("team", {}).get("name")
                away = (teams.get("away") or {}).get("team", {}).get("name")
                if not home or not away:
                    continue
                events.append(
                    _event(
                        "béisbol",
                        f"MLB {game.get('gameType', '')}".strip(),
                        day,
                        away,
                        home,
                        (teams.get("away") or {}).get("score"),
                        (teams.get("home") or {}).get("score"),
                        "mlb-statsapi",
                    )
                )
    return events, errors


def fetch_nhl(days: set[date]) -> tuple[list[dict], list[str]]:
    events, errors = [], []
    for day in days:
        payload = _json(f"{NHL}/{day.isoformat()}")
        if not isinstance(payload, dict):
            errors.append(f"nhl:{day}")
            continue
        for game in payload.get("games", []):
            state = game.get("gameState")
            if state not in {"FINAL", "OFF", "FINAL_OT", "FINAL_SO"}:
                continue
            home = (game.get("homeTeam") or {}).get("abbrev")
            away = (game.get("awayTeam") or {}).get("abbrev")
            if not home or not away:
                continue
            events.append(
                _event(
                    "hockey hielo",
                    "NHL",
                    day,
                    away,
                    home,
                    (game.get("awayTeam") or {}).get("score"),
                    (game.get("homeTeam") or {}).get("score"),
                    "nhl-api",
                )
            )
    return events, errors


def fetch_nfl(days: set[date]) -> tuple[list[dict], list[str]]:
    text = _get(NFLVERSE, timeout=40)
    if text is None:
        return [], ["nflverse"]
    events = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            day = date.fromisoformat(row.get("gameday", ""))
        except ValueError:
            continue
        if day not in days:
            continue
        hg, ag = _ft(row.get("home_score")), _ft(row.get("away_score"))
        home, away = row.get("home_team"), row.get("away_team")
        if hg is None or ag is None or not home or not away:
            continue
        events.append(_event("fútbol americano", "NFL", day, away, home, ag, hg, "nflverse"))
    return events, []


# ---------------- persistencia ----------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _merge_matches(existing: list[dict], new: list[dict]) -> list[dict]:
    by_key = {(row.get("l"), row.get("d"), row.get("h"), row.get("a")): row for row in existing}
    for row in new:
        key = (row.get("l"), row.get("d"), row.get("h"), row.get("a"))
        if key in by_key:
            if row.get("hg") is not None and row.get("ag") is not None:
                by_key[key] = row
        else:
            by_key[key] = row
    return sorted(by_key.values(), key=lambda row: (row.get("d", ""), row.get("l", "")))


def _merge_events(existing: list[dict], new: list[dict]) -> list[dict]:
    by_key = {(row.get("sport"), row.get("date"), row.get("home"), row.get("away")): row for row in existing}
    for row in new:
        key = (row.get("sport"), row.get("date"), row.get("home"), row.get("away"))
        by_key[key] = row
    return sorted(by_key.values(), key=lambda row: (row.get("date", ""), row.get("sport", "")))


def read_live_matches(path: Path = LIVE_MATCHES) -> list[dict]:
    return _read_jsonl(path)


def update(day: date | None = None, days: int = 1) -> dict:
    day = day or (date.today() - timedelta(days=1))
    dates = {day - timedelta(days=offset) for offset in range(max(1, days))}
    errors: list[str] = []

    football, err = fetch_openfootball(dates)
    errors.extend(err)
    extra, err = fetch_openligadb(dates)
    football.extend(extra)
    errors.extend(err)
    sportsdb_matches, events, err = fetch_sportsdb(dates)
    football.extend(sportsdb_matches)
    errors.extend(err)
    for fetcher in (fetch_mlb, fetch_nhl, fetch_nfl):
        items, err = fetcher(dates)
        events.extend(items)
        errors.extend(err)

    existing_matches = read_live_matches()
    existing_events = _read_jsonl(LIVE_EVENTS)
    merged_matches = _merge_matches(existing_matches, football)
    merged_events = _merge_events(existing_events, events)
    _write_jsonl(LIVE_MATCHES, merged_matches)
    _write_jsonl(LIVE_EVENTS, merged_events)

    new_matches = len(merged_matches) - len(existing_matches)
    new_events = len(merged_events) - len(existing_events)
    by_sport = Counter(row.get("sport") for row in events)
    by_source = Counter(row.get("src") or row.get("source") for row in football + events)
    return {
        "date": day.isoformat(),
        "days": sorted(item.isoformat() for item in dates),
        "football_matches": len(football),
        "events": len(events),
        "new_matches": max(new_matches, 0),
        "new_events": max(new_events, 0),
        "by_sport": dict(by_sport),
        "by_source": dict(by_source),
        "errors": sorted(set(errors)),
    }


def summary() -> dict:
    matches = read_live_matches()
    events = _read_jsonl(LIVE_EVENTS)
    return {
        "matches": len(matches),
        "events": len(events),
        "last_match_date": max((row.get("d") for row in matches), default=None),
        "last_event_date": max((row.get("date") for row in events), default=None),
        "sports": dict(Counter(row.get("sport") for row in events)),
        "sources": dict(Counter((row.get("src") or row.get("source")) for row in matches + events)),
        "latest": sorted(events, key=lambda row: row.get("date", ""), reverse=True)[:30],
    }
