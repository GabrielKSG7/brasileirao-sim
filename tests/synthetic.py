"""Gerador de ligas sintéticas com forças conhecidas, para testar os modelos sem dados reais."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def make_league(
    n_teams: int = 10,
    seasons: tuple[int, ...] = (2097, 2098),
    home: float = 0.3,
    base: float = 0.1,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """Pontos corridos (ida e volta) por temporada, uma rodada por semana.
    Devolve (jogos, ataque_real, defesa_real)."""
    rng = np.random.default_rng(seed)
    teams = [f"t{i:02d}" for i in range(n_teams)]
    attack = dict(zip(teams, np.linspace(0.5, -0.5, n_teams)))
    defense = dict(zip(teams, np.linspace(0.4, -0.4, n_teams)))
    rows = []
    for season in seasons:
        start = dt.date(season, 2, 1)
        pairs = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(pairs)
        per_round = n_teams // 2
        for k, (h, a) in enumerate(pairs):
            date = start + dt.timedelta(days=7 * (k // per_round))
            lam = np.exp(base + home + attack[h] - defense[a])
            mu = np.exp(base + attack[a] - defense[h])
            rows.append(
                {
                    "season": season,
                    "match_date": date,
                    "home_team_id": h,
                    "away_team_id": a,
                    "home_goals": int(rng.poisson(lam)),
                    "away_goals": int(rng.poisson(mu)),
                    "is_played": True,
                }
            )
    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df.sort_values(["match_date", "home_team_id"]).reset_index(drop=True), attack, defense
