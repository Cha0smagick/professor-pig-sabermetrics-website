import json

import pytest

from sabermetria import stats

MATCHES = [
    {"l": "en.1", "s": "2024-25", "d": "2024-08-01", "h": "A", "a": "B", "hg": 2, "ag": 0, "ht": [1, 0]},
    {"l": "en.1", "s": "2024-25", "d": "2024-08-02", "h": "C", "a": "D", "hg": 1, "ag": 1, "ht": [0, 1]},
    {"l": "en.1", "s": "2024-25", "d": "2024-08-03", "h": "B", "a": "C", "hg": 0, "ag": 2, "ht": [0, 0]},
    {"l": "en.1", "s": "2024-25", "d": "2024-08-04", "h": "D", "a": "A", "hg": 1, "ag": 2, "ht": [1, 1]},
    {"l": "en.1", "s": "2024-25", "d": "2024-08-05", "h": "A", "a": "C", "hg": 0, "ag": 1, "ht": [0, 1]},
    {"l": "en.1", "s": "2024-25", "d": "2024-08-06", "h": "B", "a": "D", "hg": 3, "ag": 2, "ht": [0, 2]},
]


def test_totals_separates_comebacks_from_leads_lost():
    matches = [
        {"l": "en.1", "s": "2024-25", "d": "2024-08-01", "h": "A", "a": "B", "hg": 2, "ag": 2, "ht": [2, 0]},
        {"l": "en.1", "s": "2024-25", "d": "2024-08-02", "h": "C", "a": "D", "hg": 2, "ag": 1, "ht": [0, 1]},
    ]
    totals = stats.totals(matches)
    assert totals["comebacks"] == 1
    assert totals["leads_lost"] == 2


def test_missing_halftime_in_goalless_match_is_imputed(tmp_path):
    from sabermetria import ingest

    raw = tmp_path / "raw"
    (raw / "2024-25").mkdir(parents=True)
    (raw / "2024-25" / "en.1.json").write_text(
        '{"matches":[{"date":"2024-08-01","team1":"A","team2":"B","score":{"ft":[0,0]}}]}',
        encoding="utf-8",
    )
    rows, _ = ingest.load_raw(raw)
    assert rows[0]["ht"] == [0, 0]


def test_totals_are_exact():
    totals = stats.totals(MATCHES)
    assert totals["matches"] == 6
    assert totals["goals"] == 15
    assert totals["avg_goals"] == pytest.approx(2.5)
    assert totals["home_win"] == pytest.approx(2 / 6, abs=1e-4)
    assert totals["draw"] == pytest.approx(1 / 6, abs=1e-4)
    assert totals["away_win"] == pytest.approx(3 / 6, abs=1e-4)
    assert totals["avg_goals_1h"] == pytest.approx(7 / 6, abs=1e-3)
    assert totals["avg_goals_2h"] == pytest.approx(8 / 6, abs=1e-3)
    assert totals["comebacks"] == 1
    assert totals["leads_lost"] == 2


def test_team_stats_and_form():
    lab = stats.league_lab(MATCHES)
    team = next(t for t in lab["teams"] if t["name"] == "A")
    assert team["m"] == 3
    assert team["w"] == 2 and team["l"] == 1
    assert team["pts"] == 6
    assert team["gf"] == 4 and team["ga"] == 2
    assert team["home"]["pts"] == 3
    assert team["away"]["pts"] == 3
    assert team["form"] == "LWW"
    assert team["luck"] == pytest.approx(6 - team["pyth"], abs=0.05)


def test_compact_matches_preserve_indices_and_halftime():
    lab = stats.league_lab(MATCHES)
    assert len(lab["matches"]) == 6
    for row in lab["matches"]:
        assert len(row) == 7
        assert 0 <= row[1] < len(lab["teamNames"])
        assert 0 <= row[2] < len(lab["teamNames"])
    first = lab["matches"][0]
    assert first[0] == "2024-08-01"
    assert (first[5], first[6]) == (1, 0)


def test_goals_distribution_matches_total():
    lab = stats.league_lab(MATCHES)
    assert sum(lab["goals_distribution"]) == len(MATCHES)


def test_advanced_sample_uses_cache_when_raw_missing(tmp_path):
    cache = tmp_path / "advanced.json"
    cache.write_text(json.dumps({"n": 2, "rows": [], "note": "cache"}), encoding="utf-8")
    result = stats.advanced_sample(tmp_path / "inexistente", cache)
    assert result["n"] == 2
    assert result["note"] == "cache"


def test_advanced_sample_without_any_source(tmp_path):
    result = stats.advanced_sample(tmp_path / "inexistente", tmp_path / "no-cache.json")
    assert result["n"] == 0
