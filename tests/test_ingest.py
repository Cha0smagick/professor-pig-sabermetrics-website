import json
from pathlib import Path

from sabermetria import ingest


def _write(root: Path, season: str, code: str, matches: list) -> None:
    folder = root / season
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{code}.json").write_text(json.dumps({"name": "Test League", "matches": matches}), encoding="utf-8")


def test_parse_valid_and_reject_invalid(tmp_path):
    matches = [
        {"date": "2024-08-01", "team1": "A", "team2": "B", "score": {"ft": [2, 1]}},
        {"date": "2024-08-02", "team1": "C", "team2": "D", "score": {"ft": [0, 0]}},
        {"date": "2024-08-03", "team1": "A", "team2": "C", "score": {}},
        {"date": "2024-08-04", "team1": "B", "team2": "D", "score": {"ft": [1]}},
        {"date": "not-a-date", "team1": "A", "team2": "D", "score": {"ft": [1, 1]}},
        {"date": "2024-08-05", "team1": "B", "team2": "C", "score": {"ft": [20, 0]}},
        {"date": "2024-08-06", "team1": "A", "team2": "B", "score": {"ft": [9, 9]}},
    ]
    _write(tmp_path, "2024-25", "en.1", matches)

    rows, rejected = ingest.load_raw(tmp_path)

    assert len(rows) == 3
    assert rejected["sin_marcador"] == 2
    assert rejected["campos_incompletos"] == 1
    assert rejected["marcador_fuera_de_rango"] == 1
    assert {row["h"] for row in rows} == {"A", "C"}


def test_dedupe_keeps_first_and_counts_conflict(tmp_path):
    matches = [
        {"date": "2024-08-01", "team1": "A", "team2": "B", "score": {"ft": [2, 1]}},
        {"date": "2024-08-01", "team1": "A", "team2": "B", "score": {"ft": [1, 0]}},
    ]
    _write(tmp_path, "2024-25", "en.1", matches)

    rows, rejected = ingest.load_raw(tmp_path)

    assert len(rows) == 1
    assert (rows[0]["hg"], rows[0]["ag"]) == (2, 1)
    assert rejected["duplicado"] == 1
    assert rejected["duplicado_conflicto"] == 1


def test_ignores_unknown_competition(tmp_path):
    _write(tmp_path, "2024-25", "xx.9", [{"date": "2024-08-01", "team1": "A", "team2": "B", "score": {"ft": [1, 0]}}])
    rows, _ = ingest.load_raw(tmp_path)
    assert rows == []


def test_summary_counts(tmp_path):
    _write(tmp_path, "2024-25", "en.1", [
        {"date": "2024-08-01", "team1": "A", "team2": "B", "score": {"ft": [2, 1]}},
        {"date": "2024-08-02", "team1": "B", "team2": "A", "score": {"ft": [0, 0]}},
    ])
    rows, _ = ingest.load_raw(tmp_path)
    summary = ingest.summarize(rows)
    assert summary["en.1"]["matches"] == 2
    assert summary["en.1"]["teams"] == 2
    assert summary["en.1"]["seasons"] == 1


def test_normalize_team_removes_suffixes_and_accents():
    assert ingest.normalize_team("Real Sociedad de Fútbol") == "real sociedad"
    assert ingest.normalize_team("AS Monaco FC") == "monaco"
    assert ingest.normalize_team("Brighton & Hove Albion FC") == "brighton hove albion"
    assert ingest.normalize_team("Bayern München") == "bayern munchen"


def test_canonicalize_merges_variants_of_same_club(tmp_path):
    _write(tmp_path, "2024-25", "en.1", [
        {"date": "2024-08-01", "team1": "Manchester City FC", "team2": "B", "score": {"ft": [2, 1]}},
        {"date": "2024-08-02", "team1": "Manchester City", "team2": "C", "score": {"ft": [1, 0]}},
        {"date": "2024-08-03", "team1": "B", "team2": "Manchester City FC", "score": {"ft": [0, 0]}},
        {"date": "2024-08-04", "team1": "C", "team2": "Manchester City", "score": {"ft": [1, 1]}},
    ])
    rows, rejected = ingest.load_raw(tmp_path)
    teams = {row["h"] for row in rows} | {row["a"] for row in rows}
    assert "Manchester City" not in teams
    assert "Manchester City FC" in teams
    assert rejected["equipo_contra_si_mismo"] == 0
