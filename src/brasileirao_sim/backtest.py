"""Backtest rodada a rodada, sem vazamento.

Para cada temporada e cada corte (véspera da temporada, depois a cada 10 jogos disputados):
1. ajusta cada modelo só com os jogos até o corte;
2. simula o restante da temporada (probabilidades por time e zona);
3. prevê os jogos da janela seguinte, até o próximo corte (avaliação jogo a jogo).

Cada jogo é previsto uma única vez, pelo corte imediatamente anterior a ele, como numa
atualização semanal real.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from brasileirao_sim.data import round_cutoffs, season_state, season_teams
from brasileirao_sim.models import dixon_coles, elo
from brasileirao_sim.simulate import ZONES, fit_model, rank, simulate


@dataclass(frozen=True)
class ModelSpec:
    name: str  # "dixon_coles" | "elo" | "tabela"
    dc: dixon_coles.DCConfig = field(default_factory=dixon_coles.DCConfig)
    elo: elo.EloConfig = field(default_factory=elo.EloConfig)


def final_positions(matches: pd.DataFrame, season: int) -> pd.DataFrame:
    """Classificação final (1 = campeão) pelos critérios pontos, vitórias, saldo e gols pró."""
    teams = season_teams(matches, season)
    s = matches[(matches["season"] == season) & matches["is_played"]]
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    pts, wins, gf, ga = (np.zeros((1, n)) for _ in range(4))
    for r in s.itertuples(index=False):
        h, a, x, y = idx[r.home_team_id], idx[r.away_team_id], r.home_goals, r.away_goals
        gf[0, h] += x
        ga[0, h] += y
        gf[0, a] += y
        ga[0, a] += x
        if x > y:
            pts[0, h] += 3
            wins[0, h] += 1
        elif x < y:
            pts[0, a] += 3
            wins[0, a] += 1
        else:
            pts[0, h] += 1
            pts[0, a] += 1
    pos = rank(pts, wins, gf, ga, np.random.default_rng(0))[0] + 1
    out = pd.DataFrame({"team_id": teams, "final_position": pos, "final_points": pts[0]})
    for zone, (lo, hi) in ZONES.items():
        out[f"final_{zone}"] = ((out["final_position"] >= lo) & (out["final_position"] <= hi))
    return out.sort_values("final_position").reset_index(drop=True)


def season_is_complete(matches: pd.DataFrame, season: int) -> bool:
    n = len(season_teams(matches, season))
    played = int(matches.loc[matches["season"] == season, "is_played"].sum())
    return played >= n * (n - 1) - 1  # tolera um jogo nunca disputado (ex.: 2016)


def run_season(
    matches: pd.DataFrame,
    season: int,
    specs: list[ModelSpec],
    n_sims: int = 10_000,
    seed: int = 2026,
    rounds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (probabilidades por time e corte, previsões jogo a jogo)."""
    cuts = round_cutoffs(matches, season)
    if rounds is not None:
        cuts = [c for c in cuts if c[0] in rounds]
    season_matches = matches[(matches["season"] == season) & matches["is_played"]]
    team_rows, match_rows = [], []

    for i, (k, cutoff) in enumerate(cuts):
        next_cut = cuts[i + 1][1] if i + 1 < len(cuts) else dt.date.max
        window = season_matches[
            (season_matches["match_date"] > cutoff) & (season_matches["match_date"] <= next_cut)
        ]
        state = season_state(matches, season, cutoff)
        for spec in specs:
            fitted = fit_model(spec.name, matches, state, spec.dc, spec.elo)
            if len(window):
                probs = fitted.outcome_probs(
                    window["home_team_id"].to_numpy(), window["away_team_id"].to_numpy()
                )
                w = window.reset_index(drop=True)
                gd = w["home_goals"] - w["away_goals"]
                match_rows.append(
                    pd.DataFrame(
                        {
                            "season": season,
                            "round": k,
                            "cutoff": cutoff,
                            "model": spec.name,
                            "match_date": w["match_date"],
                            "home_team_id": w["home_team_id"],
                            "away_team_id": w["away_team_id"],
                            "home_goals": w["home_goals"].astype(int),
                            "away_goals": w["away_goals"].astype(int),
                            "result": np.where(gd > 0, "H", np.where(gd == 0, "D", "A")),
                            "p_home": probs[:, 0],
                            "p_draw": probs[:, 1],
                            "p_away": probs[:, 2],
                        }
                    )
                )
            if n_sims > 0:
                rng = np.random.default_rng(seed + 1000 * season + k)
                fc = simulate(state, fitted.sample_scores, n_sims, rng, spec.name)
                z = fc.zone_probs()
                z.insert(0, "model", spec.name)
                z.insert(0, "cutoff", cutoff)
                z.insert(0, "round", k)
                z.insert(0, "season", season)
                team_rows.append(z)

    teams = pd.concat(team_rows, ignore_index=True) if team_rows else pd.DataFrame()
    games = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()
    return teams, games


def market_predictions(matches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Previsões do mercado (odd média de fechamento) para os mesmos jogos do backtest."""
    base = games[games["model"] == games["model"].iloc[0]].drop(
        columns=["model", "p_home", "p_draw", "p_away"]
    )
    mk = matches[["season", "home_team_id", "away_team_id", "market_p_home", "market_p_draw",
                  "market_p_away"]]
    out = base.merge(mk, on=["season", "home_team_id", "away_team_id"], how="left")
    out = out.rename(
        columns={"market_p_home": "p_home", "market_p_draw": "p_draw", "market_p_away": "p_away"}
    )
    out["model"] = "mercado"
    return out[games.columns]
