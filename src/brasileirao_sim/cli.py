"""Linha de comando.

    python -m brasileirao_sim.cli simulate --season 2026 --as-of 2026-11-20
    python -m brasileirao_sim.cli simulate --season 2025 --as-of 2025-08-01 --model elo
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from brasileirao_sim.data import load_matches
from brasileirao_sim.simulate import MODELS, forecast


def _pct(x: float) -> str:
    if x >= 0.9995:
        return ">99,9%"
    if 0 < x < 0.0005:
        return "<0,1%"
    return f"{x:.1%}".replace(".", ",")


def cmd_simulate(args) -> int:
    matches = load_matches(args.db)
    cutoff = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()
    dc_cfg = elo_cfg = None
    params = Path(args.params)
    if params.exists():
        from brasileirao_sim.tune import load_frozen_specs

        specs = {s.name: s for s in load_frozen_specs(params)}
        dc_cfg, elo_cfg = specs["dixon_coles"].dc, specs["elo"].elo
    fc, fitted, state = forecast(
        matches, args.season, cutoff, args.model, args.sims, args.seed, dc_cfg, elo_cfg
    )
    print(
        f"Brasileirão {args.season} | modelo: {args.model} | corte: {cutoff} | "
        f"~{state.round_estimate:.0f} rodadas disputadas | {len(state.remaining)} jogos a simular "
        f"| {args.sims} simulações"
    )
    if hasattr(fitted, "home"):
        print(f"mando de campo: {fitted.home:+.3f} (log)  rho: {getattr(fitted, 'rho', 0):+.3f}")
    table = fc.zone_probs()
    shown = table.copy()
    for col in ["titulo", "g4", "libertadores", "sul_americana", "rebaixamento"]:
        shown[col] = shown[col].map(_pct)
    shown["pontos_esperados"] = shown["pontos_esperados"].round(1)
    shown["posicao_media"] = shown["posicao_media"].round(1)
    shown.index = range(1, len(shown) + 1)
    with pd.option_context("display.width", 160):
        print(shown.to_string())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.assign(season=args.season, cutoff=cutoff, model=args.model).to_csv(out, index=False)
        print(f"salvo em {out}")
    return 0


def _parse_seasons(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_backtest(args) -> int:
    import hashlib
    import json
    import time

    from brasileirao_sim.backtest import (
        final_positions,
        market_predictions,
        run_season,
        season_is_complete,
    )
    from brasileirao_sim.tune import HOLDOUT_SEASON, PARAMS_PATH, load_frozen_specs

    matches = load_matches(args.db)
    specs = load_frozen_specs(Path(args.params))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    teams, games, finals = [], [], []
    for season in _parse_seasons(args.seasons):
        t0 = time.time()
        sims = args.sims_holdout if season == HOLDOUT_SEASON else args.sims
        t, g = run_season(matches, season, specs, n_sims=sims, seed=args.seed)
        teams.append(t)
        games.append(pd.concat([g, market_predictions(matches, g)], ignore_index=True))
        if season_is_complete(matches, season):
            finals.append(final_positions(matches, season).assign(season=season))
        print(f"{season}: {t['round'].nunique()} cortes, {len(g) // len(specs)} jogos, "
              f"{sims} simulações, {time.time() - t0:.0f}s", flush=True)

    pd.concat(teams, ignore_index=True).to_parquet(out / "team_probs.parquet", index=False)
    pd.concat(games, ignore_index=True).to_parquet(out / "match_preds.parquet", index=False)
    pd.concat(finals, ignore_index=True).to_parquet(out / "final_positions.parquet", index=False)
    manifest_path = Path(args.db).parent / "raw" / "fduk_bra.manifest.json"
    meta = {
        "gerado_em": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "temporadas": _parse_seasons(args.seasons),
        "simulacoes": args.sims,
        "simulacoes_holdout": args.sims_holdout,
        "semente": args.seed,
        "parametros": json.loads(Path(args.params).read_text(encoding="utf-8")),
        "parametros_sha256": hashlib.sha256(Path(args.params).read_bytes()).hexdigest(),
        "dados": json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else None,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"resultados em {out}/ (parâmetros de {PARAMS_PATH})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="brasileirao-sim")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="probabilidades por zona numa data de corte")
    s.add_argument("--season", type=int, required=True)
    s.add_argument("--as-of", help="data de corte AAAA-MM-DD (padrão: hoje)")
    s.add_argument("--model", choices=MODELS, default="dixon_coles")
    s.add_argument("--sims", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--db", default="data/brasileirao.duckdb")
    s.add_argument("--out", help="salva a tabela em CSV")
    s.add_argument("--params", default="config/model_params.json",
                   help="parâmetros congelados (usa os padrões do código se não existir)")
    s.set_defaults(func=cmd_simulate)

    b = sub.add_parser("backtest", help="backtest rodada a rodada com os parâmetros congelados")
    b.add_argument("--seasons", default="2019-2026", help="ex.: 2019-2026 ou 2023,2026")
    b.add_argument("--sims", type=int, default=5000, help="simulações nas temporadas de validação")
    b.add_argument("--sims-holdout", type=int, default=10_000, help="simulações em 2026")
    b.add_argument("--seed", type=int, default=2026)
    b.add_argument("--db", default="data/brasileirao.duckdb")
    b.add_argument("--params", default="config/model_params.json")
    b.add_argument("--out", default="results/backtest")
    b.set_defaults(func=cmd_backtest)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
