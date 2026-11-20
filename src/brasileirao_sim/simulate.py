"""Simulação Monte Carlo do restante do campeonato e probabilidades por zona.

Desempate (regulamento da CBF): pontos, vitórias, saldo de gols, gols pró e, na v1, sorteio
no lugar de confronto direto / cartões (raro de decidir algo).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from brasileirao_sim.data import SeasonState, as_of, season_state
from brasileirao_sim.models import dixon_coles, elo

# Zonas por posição (regulamento 2026, sem redistribuição de vagas pelas copas).
ZONES: dict[str, tuple[int, int]] = {
    "titulo": (1, 1),
    "g4": (1, 4),
    "libertadores": (1, 5),
    "sul_americana": (6, 11),
    "rebaixamento": (17, 20),
}


@dataclass(frozen=True)
class Forecast:
    season: int
    cutoff: dt.date
    model: str
    teams: tuple[str, ...]
    position_probs: np.ndarray  # (n_times, n_posições): P(time termina na posição j+1)
    expected_points: np.ndarray
    current_points: np.ndarray
    games_played: np.ndarray
    n_sims: int

    def zone_probs(self) -> pd.DataFrame:
        out = pd.DataFrame({"team_id": self.teams})
        for zone, (lo, hi) in ZONES.items():
            out[zone] = self.position_probs[:, lo - 1 : hi].sum(axis=1)
        out["pontos_atuais"] = self.current_points
        out["jogos"] = self.games_played
        out["pontos_esperados"] = self.expected_points
        out["posicao_media"] = self.position_probs @ np.arange(1, len(self.teams) + 1)
        return out.sort_values("posicao_media").reset_index(drop=True)


def _accumulate(n_teams, n_sims, hi, ai, hg, ag):
    """Pontos, vitórias e gols pró/contra por simulação. hg/ag: (n_jogos, n_sims)."""
    pts = np.zeros((n_sims, n_teams))
    wins = np.zeros((n_sims, n_teams))
    gf = np.zeros((n_sims, n_teams))
    ga = np.zeros((n_sims, n_teams))
    for k in range(len(hi)):
        h, a, x, y = hi[k], ai[k], hg[k], ag[k]
        hw, aw, dr = x > y, x < y, x == y
        pts[:, h] += 3 * hw + dr
        pts[:, a] += 3 * aw + dr
        wins[:, h] += hw
        wins[:, a] += aw
        gf[:, h] += x
        ga[:, h] += y
        gf[:, a] += y
        ga[:, a] += x
    return pts, wins, gf, ga


def rank(pts, wins, gf, ga, rng: np.random.Generator) -> np.ndarray:
    """Posição (0 = líder) de cada time em cada simulação. Entradas: (n_sims, n_times)."""
    gd = gf - ga
    key = (
        pts * 1e9
        + wins * 1e6
        + (gd + 500) * 1e3
        + np.minimum(gf, 999)
        + rng.random(pts.shape) * 0.5  # sorteio
    )
    order = np.argsort(-key, axis=1)
    positions = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    positions[rows, order] = np.arange(order.shape[1])[None, :]
    return positions


def simulate(state: SeasonState, sampler, n_sims: int, rng: np.random.Generator, model_name: str):
    teams = state.teams
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    p = state.played
    ph = p["home_team_id"].map(idx).to_numpy(dtype=np.int64)
    pa = p["away_team_id"].map(idx).to_numpy(dtype=np.int64)
    p_hg = p["home_goals"].to_numpy(dtype=np.int64)[:, None]
    p_ag = p["away_goals"].to_numpy(dtype=np.int64)[:, None]
    base = _accumulate(n, 1, ph, pa, p_hg, p_ag)

    r = state.remaining
    if len(r):
        hg, ag = sampler(r["home_team_id"].to_numpy(), r["away_team_id"].to_numpy(), n_sims, rng)
        sim = _accumulate(
            n,
            n_sims,
            r["home_team_id"].map(idx).to_numpy(dtype=np.int64),
            r["away_team_id"].map(idx).to_numpy(dtype=np.int64),
            hg,
            ag,
        )
    else:
        sim = tuple(np.zeros((n_sims, n)) for _ in range(4))

    pts, wins, gf, ga = (b + s for b, s in zip(base, sim))
    positions = rank(pts, wins, gf, ga, rng)
    probs = np.zeros((n, n))
    for j in range(n):
        probs[:, j] = (positions == j).mean(axis=0)

    games = np.bincount(ph, minlength=n) + np.bincount(pa, minlength=n)
    return Forecast(
        season=state.season,
        cutoff=state.cutoff,
        model=model_name,
        teams=teams,
        position_probs=probs,
        expected_points=pts.mean(axis=0),
        current_points=base[0][0],
        games_played=games,
        n_sims=n_sims,
    )


MODELS = ("dixon_coles", "elo", "tabela")


def fit_model(name: str, matches: pd.DataFrame, state: SeasonState, dc_config=None, elo_config=None):
    """Ajusta o modelo `name` só com o que se sabia em state.cutoff."""
    if name == "dixon_coles":
        cfg = dc_config or dixon_coles.DCConfig()
        data = as_of(matches, state.cutoff, lookback_days=cfg.lookback_days)
        return dixon_coles.fit(data, state.teams, state.promoted, cfg)
    if name == "elo":
        data = as_of(matches, state.cutoff)
        return elo.fit_elo(data, state.teams, state.season, state.promoted,
                           elo_config or elo.EloConfig())
    if name == "tabela":
        data = as_of(matches, state.cutoff)
        return elo.fit_table(data, state.teams, state.season)
    raise ValueError(f"modelo desconhecido: {name}. Opções: {MODELS}")


def forecast(
    matches: pd.DataFrame,
    season: int,
    cutoff: dt.date,
    model: str = "dixon_coles",
    n_sims: int = 10_000,
    seed: int = 2026,
    dc_config=None,
    elo_config=None,
) -> tuple[Forecast, object, SeasonState]:
    state = season_state(matches, season, cutoff)
    fitted = fit_model(model, matches, state, dc_config, elo_config)
    rng = np.random.default_rng(seed)
    fc = simulate(state, fitted.sample_scores, n_sims, rng, model)
    return fc, fitted, state
