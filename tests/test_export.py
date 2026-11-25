import json
from pathlib import Path

from sabermetria import export


def _season(root: Path, season: str, offset: int) -> None:
    teams = [f"Team{i}" for i in range(6)]
    matches = []
    day = 1
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if home == away:
                continue
            matches.append(
                {
                    "date": f"20{24 + offset}-{1 + (day // 28) % 12:02d}-{1 + day % 28:02d}",
                    "team1": home,
                    "team2": away,
                    "score": {"ft": [(i + 1) % 4, (j + offset) % 3]},
                }
            )
            day += 1
    folder = root / season
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "en.1.json").write_text(json.dumps({"name": "Test", "matches": matches}), encoding="utf-8")


def test_build_writes_valid_schema(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    for index, season in enumerate(("2021-22", "2022-23", "2023-24", "2024-25")):
        _season(raw, season, index)

    result = export.build(raw, tmp_path / "cache.jsonl.gz", out, tmp_path / "aliases.json", extra=None)

    assert result["matches"] == 120
    leagues = json.loads((out / "leagues.json").read_text(encoding="utf-8"))
    ratings = json.loads((out / "ratings.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))

    assert leagues[0]["code"] == "en.1"
    assert "validated" in leagues[0]
    assert ratings["sport"] == "football"
    assert len(ratings["leagues"]["en.1"]["teams"]) == 6
    assert ratings["leagues"]["en.1"]["alpha"] >= 0
    assert len(ratings["leagues"]["en.1"]["base"]) == 3
    assert metrics["aggregate"]["n"] > 0
    assert metrics["leagues"]["en.1"]["n"] > 0
    assert metrics["totals"]["matches"] == 120
    assert "aggregate_validated" in metrics
    assert len(metrics["aggregate"]["reliability"]) == 10


def test_build_is_deterministic(tmp_path):
    raw = tmp_path / "raw"
    for index, season in enumerate(("2021-22", "2022-23", "2023-24", "2024-25")):
        _season(raw, season, index)
    cache = tmp_path / "cache.jsonl.gz"

    first = tmp_path / "first"
    second = tmp_path / "second"
    export.build(raw, cache, first, tmp_path / "aliases.json", extra=None)
    export.build(raw, cache, second, tmp_path / "aliases.json", extra=None)

    for name in ("leagues.json", "ratings.json"):
        one = json.loads((first / name).read_text(encoding="utf-8"))
        two = json.loads((second / name).read_text(encoding="utf-8"))
        if isinstance(one, dict):
            one.pop("generated_at", None)
            two.pop("generated_at", None)
        assert one == two
