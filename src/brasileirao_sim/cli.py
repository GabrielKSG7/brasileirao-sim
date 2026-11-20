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
    fc, fitted, state = forecast(matches, args.season, cutoff, args.model, args.sims, args.seed)
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
    s.set_defaults(func=cmd_simulate)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
