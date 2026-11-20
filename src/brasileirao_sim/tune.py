"""Ajuste de hiperparâmetros SÓ com temporadas de validação (2019–2025).

Etapa 1 (jogo a jogo): grade de parâmetros do Dixon-Coles e do Elo, escolhendo a menor
log loss média das previsões da janela seguinte a cada corte.
Etapa 2 (temporada): com os melhores parâmetros da etapa 1, escolhe o tamanho do ruído de
força na simulação pelo Brier médio das probabilidades por zona.

Os parâmetros escolhidos vão para config/model_params.json e ficam CONGELADOS para o
backtest de 2026, que não participa de nada aqui.

    python -m brasileirao_sim.tune --stage matches
    python -m brasileirao_sim.tune --stage season
"""

from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from brasileirao_sim.backtest import ModelSpec, final_positions, run_season
from brasileirao_sim.data import load_matches
from brasileirao_sim.metrics import binary_scores, match_scores
from brasileirao_sim.models.dixon_coles import DCConfig
from brasileirao_sim.models.elo import EloConfig
from brasileirao_sim.simulate import ZONES

VALIDATION_SEASONS = tuple(range(2019, 2026))
HOLDOUT_SEASON = 2026
PARAMS_PATH = Path("config/model_params.json")
OUT_DIR = Path("results/tuning")

DC_GRID = {
    "xi": [0.001, 0.0019, 0.003, 0.0045],
    "ridge": [2.0, 5.0, 10.0, 20.0],
    "promoted": [0.0, -0.1, -0.2, -0.3],
}
# Grade 2: a primeira (k 15-30, regressão 0.2-0.5, promovidos -50 a -150) teve o ótimo na
# borda em k, regressão e promovidos; esta estende nessas direções (grade 1 em
# results/tuning/elo_jogos_grade1.csv).
ELO_GRID = {
    "k": [8.0, 11.0, 15.0],
    "home": [70.0, 90.0],
    "season_regress": [0.0, 0.1, 0.2],
    "promoted_offset": [0.0, -25.0, -50.0],
}
# Grade 2 (a grade 1 ia até 0.2 e o ótimo caiu na borda; ver ruido_temporada_grade1.csv)
DC_SD_GRID = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
ELO_SD_GRID = [0.0, 25.0, 50.0, 75.0, 100.0]

_MATCHES: pd.DataFrame | None = None


def _init(db: str) -> None:
    global _MATCHES
    _MATCHES = load_matches(db)


def _match_logloss(spec: ModelSpec) -> dict:
    frames = [run_season(_MATCHES, s, [spec], n_sims=0)[1] for s in VALIDATION_SEASONS]
    g = pd.concat(frames, ignore_index=True)
    sc = match_scores(g[["p_home", "p_draw", "p_away"]].to_numpy(), g["result"])
    return {"n": len(g), **sc.mean().to_dict()}


def _season_brier(args: tuple[ModelSpec, int]) -> dict:
    spec, n_sims = args
    rows = []
    for s in VALIDATION_SEASONS:
        teams, _ = run_season(_MATCHES, s, [spec], n_sims=n_sims)
        final = final_positions(_MATCHES, s)
        rows.append(teams.merge(final, on="team_id"))
    df = pd.concat(rows, ignore_index=True)
    out = {}
    for zone in ZONES:
        sc = binary_scores(df[zone].to_numpy(), df[f"final_{zone}"].to_numpy())
        out[f"brier_{zone}"] = sc["brier"].mean()
        out[f"logloss_{zone}"] = sc["log_loss"].mean()
    out["brier_medio"] = np.mean([out[f"brier_{z}"] for z in ZONES])
    out["logloss_medio"] = np.mean([out[f"logloss_{z}"] for z in ZONES])
    return out


def _dc_specs() -> list[ModelSpec]:
    specs = []
    for xi, ridge, prom in itertools.product(*DC_GRID.values()):
        cfg = DCConfig(xi=xi, ridge=ridge, promoted_attack=prom, promoted_defense=prom)
        specs.append(ModelSpec("dixon_coles", dc=cfg))
    return specs


def _elo_specs() -> list[ModelSpec]:
    return [
        ModelSpec("elo", elo=EloConfig(k=k, home=h, season_regress=r, promoted_offset=p))
        for k, h, r, p in itertools.product(*ELO_GRID.values())
    ]


def _run(fn, items, db, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(db,)) as ex:
        return list(ex.map(fn, items))


def stage_matches(db: str, workers: int, only: str | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    todo = [("dixon_coles", _dc_specs), ("elo", _elo_specs)]
    for name, make_specs in todo:
        if only and name != only:
            csv = OUT_DIR / f"{name}_jogos.csv"
            result[name] = pd.read_csv(csv).iloc[0].to_dict()
            continue
        specs = make_specs()
        scores = _run(_match_logloss, specs, db, workers)
        cfgs = [asdict(s.dc if name == "dixon_coles" else s.elo) for s in specs]
        table = pd.DataFrame([{**c, **sc} for c, sc in zip(cfgs, scores)])
        table = table.sort_values("log_loss").reset_index(drop=True)
        table.to_csv(OUT_DIR / f"{name}_jogos.csv", index=False)
        result[name] = table.iloc[0].to_dict()
        print(f"\n{name}: 5 melhores de {len(table)}")
        print(table.head(5).to_string())
    params = _load_params()
    params["dixon_coles"] = {
        k: type(v)(result["dixon_coles"][k]) for k, v in asdict(DCConfig()).items()
    }
    params["elo"] = {k: type(v)(result["elo"][k]) for k, v in asdict(EloConfig()).items()}
    params["validacao"] = {
        "temporadas": list(VALIDATION_SEASONS),
        "objetivo_etapa1": "log loss média dos jogos da janela seguinte a cada corte",
        "dixon_coles_log_loss": result["dixon_coles"]["log_loss"],
        "elo_log_loss": result["elo"]["log_loss"],
    }
    _save_params(params)
    return params


def stage_season(db: str, workers: int, n_sims: int) -> dict:
    params = _load_params()
    dc = DCConfig(**params["dixon_coles"])
    el = EloConfig(**params["elo"])
    items = [(ModelSpec("dixon_coles", dc=replace(dc, sim_strength_sd=sd)), n_sims)
             for sd in DC_SD_GRID]
    items += [(ModelSpec("elo", elo=replace(el, sim_rating_sd=sd)), n_sims) for sd in ELO_SD_GRID]
    scores = _run(_season_brier, items, db, workers)
    rows = []
    for (spec, _), sc in zip(items, scores):
        sd = spec.dc.sim_strength_sd if spec.name == "dixon_coles" else spec.elo.sim_rating_sd
        rows.append({"model": spec.name, "sd": sd, **sc})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "ruido_temporada.csv", index=False)
    print(table[["model", "sd", "brier_medio", "logloss_medio", "brier_titulo",
                 "brier_rebaixamento"]].to_string())
    for name, key in (("dixon_coles", "sim_strength_sd"), ("elo", "sim_rating_sd")):
        best = table[table["model"] == name].sort_values("brier_medio").iloc[0]
        params[name][key] = float(best["sd"])
    params["validacao"]["objetivo_etapa2"] = (
        f"Brier médio das zonas {list(ZONES)} em todos os cortes ({n_sims} simulações)"
    )
    _save_params(params)
    return params


def _load_params() -> dict:
    if PARAMS_PATH.exists():
        return json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_params(params: dict) -> None:
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(params, default=float))
    PARAMS_PATH.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"parâmetros salvos em {PARAMS_PATH}")


def load_frozen_specs(path: Path = PARAMS_PATH) -> list[ModelSpec]:
    """Especificações congeladas usadas no backtest de 2026 e nas atualizações de 2027."""
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ModelSpec("dixon_coles", dc=DCConfig(**params["dixon_coles"])),
        ModelSpec("elo", elo=EloConfig(**params["elo"])),
        ModelSpec("tabela"),
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ajuste de hiperparâmetros em 2019–2025.")
    p.add_argument("--stage", choices=["matches", "season"], required=True)
    p.add_argument("--db", default="data/brasileirao.duckdb")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--sims", type=int, default=2000)
    p.add_argument("--only", choices=["dixon_coles", "elo"], help="refaz só um modelo")
    args = p.parse_args(argv)
    if args.stage == "matches":
        stage_matches(args.db, args.workers, args.only)
    else:
        stage_season(args.db, args.workers, args.sims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
