"""CLI: python -m sabermetria {ingest,build,backtest,predict}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import backtest, export, ingest, ratings


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass


def cmd_ingest(args) -> int:
    matches, rejected = ingest.load_raw(Path(args.raw), Path(args.aliases), Path(args.extra))
    path = ingest.write_cache(matches, Path(args.cache))
    summary = ingest.summarize(matches)
    print(f"ligas={len(summary)} partidos={len(matches)} rechazados={dict(rejected)}")
    print(f"cache={path} ({path.stat().st_size / 1024:.0f} KB)")
    for code, info in sorted(summary.items()):
        print(f"  {code:8} {info['matches']:6} partidos  {info['seasons']:2} temp  {info['teams']:3} equipos")
    return 0


def cmd_build(args) -> int:
    print("Construyendo motor y artefactos web...")
    result = export.build(
        Path(args.raw), Path(args.cache), Path(args.out), Path(args.aliases), Path(args.extra)
    )
    agg = result["aggregate"]
    print("")
    print(f"ligas={result['leagues']} partidos={result['matches']} rechazados={result['rejected']}")
    if agg.get("n"):
        print(f"backtest n={agg['n']}  Brier={agg['brier']:.4f}  LogLoss={agg['logloss']:.4f}  Acc={agg['accuracy'] * 100:.1f}%")
        print(f"baseline  Brier={agg['base_brier']:.4f}  LogLoss={agg['base_logloss']:.4f}  Acc={agg['base_accuracy'] * 100:.1f}%")
    return 0


def cmd_backtest(args) -> int:
    matches = ingest.ingest(Path(args.raw), Path(args.cache), Path(args.aliases), Path(args.extra))[0]
    groups: dict[str, list[dict]] = {}
    for match in matches:
        groups.setdefault(match["l"], []).append(match)
    results = {}
    for code, rows in sorted(groups.items()):
        try:
            results[code] = backtest.evaluate_league(rows)
        except ValueError as error:
            print(f"{code}: {error}")
    agg = backtest.aggregate(results)
    print(json.dumps({k: v for k, v in agg.items() if k != "leagues"}, indent=2, ensure_ascii=False))
    return 0


def cmd_predict(args) -> int:
    matches = ingest.ingest(Path(args.raw), Path(args.cache), Path(args.aliases), Path(args.extra))[0]
    rows = [m for m in matches if m["l"] == args.league]
    if not rows:
        print(f"Liga desconocida o sin datos: {args.league}")
        return 2
    model = ratings.fit(rows)
    matrix = ratings.score_matrix(model, args.home, args.away, neutral=args.neutral)
    from .model import full_report

    report = full_report(matrix)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8()
    parser = argparse.ArgumentParser(prog="sabermetria", description="Motor predictivo de futbol")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--raw", default=str(ingest.DEFAULT_RAW))
    common.add_argument("--cache", default=str(ingest.DEFAULT_CACHE))
    common.add_argument("--aliases", default=str(ingest.DEFAULT_ALIASES))
    common.add_argument("--extra", default=str(ingest.DEFAULT_EXTRA))

    p_ingest = sub.add_parser("ingest", parents=[common], help="normaliza los JSON de openfootball")
    p_ingest.set_defaults(func=cmd_ingest)

    p_build = sub.add_parser("build", parents=[common], help="entrena, valida y exporta la web")
    p_build.add_argument("--out", default=str(export.OUT_DIR))
    p_build.set_defaults(func=cmd_build)

    p_back = sub.add_parser("backtest", parents=[common], help="solo metricas walk-forward")
    p_back.set_defaults(func=cmd_backtest)

    p_pred = sub.add_parser("predict", parents=[common], help="predice un partido")
    p_pred.add_argument("--league", required=True)
    p_pred.add_argument("--home", required=True)
    p_pred.add_argument("--away", required=True)
    p_pred.add_argument("--neutral", action="store_true")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
