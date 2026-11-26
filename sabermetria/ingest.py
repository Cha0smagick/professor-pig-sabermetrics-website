"""Ingesta: openfootball (JSON) -> partidos normalizados.

Esquema de salida (JSONL, claves cortas para reducir tamano):
    l  liga (codigo)      s  temporada        d  fecha ISO
    h  local              a  visitante        hg goles local    ag goles visitante

Reglas duras:
- Solo partidos con marcador final de dos enteros 0..15.
- Fecha ISO parseable.
- Deduplicado por (liga, fecha, local, visitante); si hay conflicto de
  marcador se conserva el primero y se cuenta.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .leagues import LEAGUES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "openfootball_data" / "football.json"
DEFAULT_EXTRA = ROOT / "datos" / "raw"
DEFAULT_LIVE = ROOT / "data" / "live_matches.jsonl"
DEFAULT_CACHE = ROOT / "data" / "matches.jsonl.gz"
DEFAULT_ALIASES = ROOT / "data" / "team_aliases.json"

MAX_GOALS = 15

# Sufijos societarios y ruido que no distingue equipos dentro de una liga.
# Eliminarlos unifica variantes del mismo club entre temporadas
# ("Manchester City" / "Manchester City FC") sin fusionar clubes distintos.
SUFFIXES = frozenset(
    {
        "fc", "afc", "cf", "sc", "ac", "club", "cd", "ud", "rc", "sv", "vfl",
        "vfb", "bsc", "fsv", "spvgg", "calcio", "ss", "ssc", "as", "us", "de",
        "futbol", "the", "cp", "fk", "nk", "sk", "if", "bk", "ff",
    }
)
ALIASES = {"utd": "united", "ath": "athletic", "athl": "athletic"}

# Fusiones curadas: (nombre canónico, variantes observadas). Solo se incluyen
# casos verificados como el mismo club. NUNCA se fusiona por parecido automatico:
# Barcelona/Espanyol, Roma/Lazio o Dundee/Dundee United son clubes distintos.
MERGE_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "co.1": [
        ("Atlético Junior", ["Junior", "Atlético Junior"]),
        ("Alianza Valledupar", ["Alianza Valledupar", "Alianza FC"]),
        ("Atlético Bucaramanga", ["Bucaramanga", "Atlético Bucaramanga"]),
        ("Boyacá Chicó", ["Chico", "Boyacá Chicó"]),
        ("Atlético Nacional", ["Atlético Nacional", "Atletico Nacional"]),
        ("América de Cali", ["América de Cali", "America de Cali"]),
        ("Independiente Medellín", ["Independiente Medellín", "Independiente Medellin"]),
        ("Águilas Doradas", ["Águilas Doradas", "Aguilas Doradas"]),
        ("Cortuluá", ["Cortuluá", "Cortulua"]),
        ("Internacional de Bogotá", ["Internacional de Bogotá", "Internacional de Bogota"]),
        ("Unión Magdalena", ["Unión Magdalena", "Union Magdalena"]),
    ],
    "es.1": [
        ("RC Celta de Vigo", ["RC Celta", "RC Celta de Vigo"]),
        ("Rayo Vallecano de Madrid", ["Rayo Vallecano", "Rayo Vallecano de Madrid"]),
        ("Real Betis Balompié", ["Real Betis", "Real Betis Balompié"]),
        ("Deportivo Alavés", ["CD Alavés", "Deportivo Alavés"]),
        ("RCD Espanyol de Barcelona", ["Espanyol Barcelona", "RCD Espanyol de Barcelona"]),
    ],
    "de.1": [
        ("TSG 1899 Hoffenheim", ["1899 Hoffenheim", "TSG 1899 Hoffenheim"]),
        ("FC St. Pauli", ["FC St. Pauli", "FC St. Pauli 1910"]),
        ("Bayer 04 Leverkusen", ["Bayer Leverkusen", "Bayer 04 Leverkusen"]),
        ("SpVgg Greuther Fürth", ["SpVgg Greuther Fürth", "SpVgg Greuther Fürth 1903"]),
    ],
    "it.1": [
        ("UC Sampdoria", ["Sampdoria", "UC Sampdoria"]),
        ("SS Lazio", ["Lazio Roma", "SS Lazio"]),
        ("Parma Calcio 1913", ["Parma FC", "Parma Calcio 1913"]),
        ("Atalanta BC", ["Atalanta", "Atalanta BC"]),
        ("Bologna FC 1909", ["Bologna FC", "Bologna FC 1909"]),
    ],
    "it.2": [
        ("FC Bari 1908", ["SSC Bari", "FC Bari 1908"]),
        ("Ascoli Calcio", ["Ascoli Picchio FC", "Ascoli Calcio"]),
        ("Reggina 1914", ["Reggina Calcio", "Reggina 1914"]),
    ],
    "fr.1": [
        ("RC Lens", ["RC Lens", "Racing Club de Lens"]),
        ("Stade Rennais FC 1901", ["Stade Rennais", "Stade Rennais FC 1901"]),
        ("RC Strasbourg Alsace", ["RC Strasbourg", "RC Strasbourg Alsace"]),
    ],
    "nl.1": [
        ("PSV Eindhoven", ["PSV", "PSV Eindhoven"]),
        ("AZ Alkmaar", ["AZ", "AZ Alkmaar"]),
        ("Feyenoord Rotterdam", ["Feyenoord", "Feyenoord Rotterdam"]),
        ("NEC Nijmegen", ["NEC", "NEC Nijmegen"]),
        ("FC Twente", ["FC Twente", "FC Twente '65"]),
        ("Willem II Tilburg", ["Willem II", "Willem II Tilburg"]),
        ("SBV Vitesse", ["Vitesse", "SBV Vitesse"]),
    ],
    "br.1": [
        ("SE Palmeiras", ["Palmeiras", "SE Palmeiras"]),
        ("Cruzeiro EC", ["Cruzeiro", "Cruzeiro EC"]),
        ("Fluminense FC", ["Fluminense RJ", "Fluminense FC"]),
        ("Sport Recife", ["Sport Recife PE", "SC Recife"]),
        ("Ceará SC", ["Ceará CE", "Ceará SC"]),
    ],
    "at.1": [("LASK Linz", ["LASK", "LASK Linz"])],
    "cn.1": [("Henan FC", ["Henan Jianye", "Henan FC"])],
    "tr.1": [
        ("Konyaspor", ["Atiker Konyaspor", "Konyaspor"]),
        ("Gaziantep FK", ["Gazisehir Gaziantep FK", "Gaziantep FK"]),
    ],
    "pt.1": [
        ("Sporting Braga", ["Sporting Braga", "Sporting Clube de Braga"]),
        ("Sporting CP", ["Sporting CP", "Sporting Clube de Portugal"]),
        ("GD Estoril Praia", ["GD Estoril", "GD Estoril Praia"]),
    ],
    "en.3": [
        ("Exeter City", ["Exeter City", "Exeter"]),
        ("Peterborough United", ["Peterborough United", "Peterborough"]),
        ("Leyton Orient", ["Leyton Orient", "Leyton"]),
        ("Plymouth Argyle", ["Plymouth Argyle", "Plymouth"]),
        ("Huddersfield Town", ["Huddersfield Town", "Huddersfield"]),
        ("Wycombe Wanderers", ["Wycombe Wanderers", "Wycombe"]),
        ("Doncaster Rovers", ["Doncaster Rovers", "Doncaster"]),
        ("Bradford City", ["Bradford City", "Bradford"]),
        ("Rotherham United", ["Rotherham United", "Rotherham"]),
        ("Burton Albion", ["Burton Albion", "Burton"]),
        ("Wigan Athletic", ["Wigan Athletic", "Wigan"]),
        ("Luton Town", ["Luton Town", "Luton"]),
        ("Lincoln City", ["Lincoln City", "Lincoln"]),
        ("Bolton Wanderers", ["Bolton Wanderers", "Bolton"]),
        ("Northampton Town", ["Northampton Town", "Northampton"]),
        ("Stockport County", ["Stockport County", "Stockport"]),
        ("Mansfield Town", ["Mansfield Town", "Mansfield"]),
        ("Cardiff City", ["Cardiff City", "Cardiff"]),
    ],
}


LEAGUE_ALIASES: dict[str, dict[str, str]] = {}
LEAGUE_DISPLAY: dict[str, dict[str, str]] = {}


def normalize_team(name: str) -> str:
    text = "".join(
        char for char in unicodedata.normalize("NFD", name) if unicodedata.category(char) != "Mn"
    ).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [ALIASES.get(token, token) for token in text.split() if token not in SUFFIXES]
    return " ".join(tokens) or text.strip()


def _build_alias_maps() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    alias_map: dict[str, dict[str, str]] = {}
    display_map: dict[str, dict[str, str]] = {}
    for league, groups in MERGE_GROUPS.items():
        alias_map[league] = {}
        display_map[league] = {}
        for display, variants in groups:
            key = normalize_team(display)
            display_map[league][key] = display
            for variant in variants:
                alias_map[league][normalize_team(variant)] = key
    return alias_map, display_map


LEAGUE_ALIASES, LEAGUE_DISPLAY = _build_alias_maps()


def _valid_date(value: str) -> str | None:
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        return None


def _halftime(score: dict | None) -> list[int] | None:
    ht = (score or {}).get("ht")
    if isinstance(ht, list) and len(ht) == 2:
        try:
            home, away = int(ht[0]), int(ht[1])
        except (TypeError, ValueError):
            return None
        if 0 <= home <= MAX_GOALS and 0 <= away <= MAX_GOALS:
            return [home, away]
    return None


def _parse(raw: Path, rejected: Counter) -> list[dict]:
    matches: list[dict] = []
    for path in sorted(raw.glob("*/*.json")):
        code = path.stem
        if code not in LEAGUES:
            continue
        season = path.parent.name
        if not re.fullmatch(r"\d{4}(-\d{2})?", season):
            rejected["temporada_invalida"] += 1
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rejected["archivo_ilegible"] += 1
            continue
        for item in payload.get("matches", []):
            score = (item.get("score") or {}).get("ft")
            if not (isinstance(score, list) and len(score) == 2):
                rejected["sin_marcador"] += 1
                continue
            try:
                hg, ag = int(score[0]), int(score[1])
            except (TypeError, ValueError):
                rejected["marcador_no_entero"] += 1
                continue
            if not (0 <= hg <= MAX_GOALS and 0 <= ag <= MAX_GOALS):
                rejected["marcador_fuera_de_rango"] += 1
                continue
            day = _valid_date(item.get("date"))
            home, away = item.get("team1"), item.get("team2")
            if not day or not home or not away:
                rejected["campos_incompletos"] += 1
                continue
            ht = _halftime(item.get("score"))
            imputed = False
            # Un 0-0 final implica 0-0 al descanso: imputarlo evita sesgar
            # los goles por parte hacia arriba (los 0-0 sin ht no son aleatorios).
            if ht is None and hg == 0 and ag == 0:
                ht = [0, 0]
                imputed = True
            matches.append(
                {
                    "l": code,
                    "s": season,
                    "d": day,
                    "h": home,
                    "a": away,
                    "hg": hg,
                    "ag": ag,
                    **({"ht": ht} if ht else {}),
                    **({"hti": True} if imputed else {}),
                }
            )
    return matches


def _parse_api_football(folder: Path, rejected: Counter) -> list[dict]:
    """Cache local de API-Football: Primera A de Colombia 2022-2024.

    Aporta marcador de medio tiempo, árbitro y sede. Solo partidos finalizados.
    """
    matches: list[dict] = []
    bogota = timezone(timedelta(hours=-5))
    for path in sorted((folder / "api_football").glob("*/fixtures.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rejected["api_ilegible"] += 1
            continue
        for item in payload.get("response", []):
            status = (item.get("fixture", {}).get("status") or {}).get("short")
            if status != "FT":
                rejected["api_no_final"] += 1
                continue
            goals = item.get("goals") or {}
            hg, ag = goals.get("home"), goals.get("away")
            if not isinstance(hg, int) or not isinstance(ag, int):
                rejected["api_sin_marcador"] += 1
                continue
            if not (0 <= hg <= MAX_GOALS and 0 <= ag <= MAX_GOALS):
                rejected["api_marcador_fuera_de_rango"] += 1
                continue
            home = (item.get("teams", {}).get("home") or {}).get("name")
            away = (item.get("teams", {}).get("away") or {}).get("name")
            raw_date = item.get("fixture", {}).get("date")
            if not home or not away or not raw_date:
                rejected["api_campos_incompletos"] += 1
                continue
            try:
                day = datetime.fromisoformat(raw_date).astimezone(bogota).date().isoformat()
            except ValueError:
                rejected["api_fecha_invalida"] += 1
                continue
            ht = (item.get("score") or {}).get("halftime") or {}
            half = None
            imputed = False
            if isinstance(ht.get("home"), int) and isinstance(ht.get("away"), int):
                half = [ht["home"], ht["away"]]
            elif hg == 0 and ag == 0:
                half = [0, 0]
                imputed = True
            record = {
                "l": "co.1",
                "s": str(item.get("league", {}).get("season", "")),
                "d": day,
                "h": home,
                "a": away,
                "hg": hg,
                "ag": ag,
                "src": "api-football",
            }
            if half:
                record["ht"] = half
            if imputed:
                record["hti"] = True
            referee = item.get("fixture", {}).get("referee")
            venue = (item.get("fixture", {}).get("venue") or {}).get("name")
            if referee:
                record["ref"] = referee
            if venue:
                record["venue"] = venue
            matches.append(record)
    return matches


def _dedupe(matches: list[dict], rejected: Counter) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for match in matches:
        key = (match["l"], match["d"], match["h"], match["a"])
        if key in seen:
            rejected["duplicado"] += 1
            if (seen[key]["hg"], seen[key]["ag"]) != (match["hg"], match["ag"]):
                rejected["duplicado_conflicto"] += 1
            continue
        seen[key] = match
    return sorted(seen.values(), key=lambda m: (m["l"], m["d"]))


def _canonicalize(matches: list[dict], rejected: Counter) -> tuple[list[dict], dict]:
    """Unifica variantes del mismo club dentro de cada liga."""
    aliases: dict[str, dict[str, str]] = {}
    for league in sorted({match["l"] for match in matches}):
        rows = [match for match in matches if match["l"] == league]
        counts = Counter([row["h"] for row in rows] + [row["a"] for row in rows])

        def key_for(name: str) -> str:
            base = normalize_team(name)
            return LEAGUE_ALIASES.get(league, {}).get(base, base)

        groups: dict[str, list[str]] = defaultdict(list)
        for name in counts:
            groups[key_for(name)].append(name)
        mapping: dict[str, str] = {}
        for key, variants in groups.items():
            display = LEAGUE_DISPLAY.get(league, {}).get(key) or max(
                variants, key=lambda name: (counts[name], name)
            )
            for variant in variants:
                mapping[variant] = display
        aliases[league] = {name: target for name, target in mapping.items() if name != target}

    rewritten, kept = [], []
    for match in matches:
        home = aliases.get(match["l"], {}).get(match["h"], match["h"])
        away = aliases.get(match["l"], {}).get(match["a"], match["a"])
        if home == away:
            rejected["equipo_contra_si_mismo"] += 1
            continue
        kept.append({**match, "h": home, "a": away})
    return kept, aliases


def load_raw(
    raw: Path = DEFAULT_RAW,
    aliases_out: Path | None = None,
    extra_dir: Path | None = None,
    live: Path | None = None,
) -> tuple[list[dict], Counter]:
    rejected: Counter = Counter()
    matches = _parse(raw, rejected)
    if extra_dir is not None:
        matches.extend(_parse_api_football(Path(extra_dir), rejected))
    if live is not None and Path(live).exists():
        matches.extend(read_live(live))
    matches = _dedupe(matches, rejected)
    matches, aliases = _canonicalize(matches, rejected)
    matches = _dedupe(matches, rejected)
    if aliases_out is not None:
        aliases_out.parent.mkdir(parents=True, exist_ok=True)
        aliases_out.write_text(json.dumps(aliases, ensure_ascii=False, indent=1), encoding="utf-8")
    return matches, rejected


def read_live(path: Path = DEFAULT_LIVE) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not all(row.get(key) for key in ("l", "d", "h", "a")):
            continue
        if row.get("hg") is None or row.get("ag") is None:
            continue
        rows.append(row)
    return rows


def read_cache(path: Path = DEFAULT_CACHE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No hay datos normalizados en {path}. Ejecuta: python -m sabermetria ingest"
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_cache(matches: list[dict], path: Path = DEFAULT_CACHE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        for match in matches:
            handle.write(json.dumps(match, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def ingest(
    raw: Path = DEFAULT_RAW,
    cache: Path = DEFAULT_CACHE,
    aliases: Path = DEFAULT_ALIASES,
    extra: Path | None = DEFAULT_EXTRA,
    live: Path | None = DEFAULT_LIVE,
) -> tuple[list[dict], Counter]:
    """Normaliza y persiste. Si el crudo no existe, usa la cache + resultados diarios."""
    if raw.exists():
        matches, rejected = load_raw(raw, aliases, extra, live)
        write_cache(matches, cache)
        return matches, rejected
    rejected: Counter = Counter()
    matches = read_cache(cache)
    if live is not None and Path(live).exists():
        matches.extend(read_live(live))
    matches = _dedupe(matches, rejected)
    matches, _ = _canonicalize(matches, rejected)
    matches = _dedupe(matches, rejected)
    return matches, rejected


def summarize(matches: list[dict]) -> dict:
    by_league: dict[str, dict] = {}
    for match in matches:
        entry = by_league.setdefault(
            match["l"], {"matches": 0, "seasons": set(), "teams": set(), "first": match["d"], "last": match["d"]}
        )
        entry["matches"] += 1
        entry["seasons"].add(match["s"])
        entry["teams"].update((match["h"], match["a"]))
        entry["first"] = min(entry["first"], match["d"])
        entry["last"] = max(entry["last"], match["d"])
    return {
        code: {
            "matches": item["matches"],
            "seasons": len(item["seasons"]),
            "teams": len(item["teams"]),
            "first": item["first"],
            "last": item["last"],
        }
        for code, item in by_league.items()
    }
