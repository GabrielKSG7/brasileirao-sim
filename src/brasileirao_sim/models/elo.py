"""Elo (baseline) e baseline "tabela atual".

Elo: ratings atualizados jogo a jogo até o corte, com vantagem de mando, multiplicador por
saldo de gols (como no World Football Elo) e regressão parcial à média no início de cada
temporada. Para virar probabilidade de vitória/empate/derrota, um logit ordenado é ajustado
sobre a diferença de rating, usando só os jogos até o corte.

Tabela atual: cada jogo usa as taxas de vitória/empate/derrota dos dois times NA TEMPORADA
até o corte, suavizadas pelas taxas históricas da liga. É o "se continuar assim" do torcedor.

Os dois baselines não produzem placar. Para a simulação, vitória vira 1x0 e empate 0x0:
pontos e vitórias ficam corretos, o saldo de gols não (afeta só o desempate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from brasileirao_sim.data import TrainingData


@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0
    home: float = 60.0              # vantagem do mandante em pontos de rating
    season_regress: float = 0.25    # fração puxada para a média a cada nova temporada
    promoted_offset: float = -80.0  # rating inicial de quem estreia = média + offset
    initial: float = 1500.0
    sim_rating_sd: float = 0.0      # incerteza do rating na simulação (pontos de Elo)


def _gd_mult(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


@dataclass(frozen=True)
class EloModel:
    ratings: dict[str, float]
    beta: float
    c1: float
    c2: float
    config: EloConfig = field(repr=False)

    def outcome_probs(self, home_ids, away_ids) -> np.ndarray:
        dr = np.array(
            [self.ratings[h] + self.config.home - self.ratings[a] for h, a in zip(home_ids, away_ids)]
        )
        return _ordered_probs(dr, self.beta, self.c1, self.c2)

    def sample_scores(self, home_ids, away_ids, n_sims: int, rng: np.random.Generator):
        sd = self.config.sim_rating_sd
        if sd <= 0:
            return _sample_from_outcomes(self.outcome_probs(home_ids, away_ids), n_sims, rng)
        teams = sorted(set(home_ids) | set(away_ids))
        pos = {t: i for i, t in enumerate(teams)}
        base = np.array([self.ratings[t] for t in teams])[:, None]
        r = base + rng.normal(0, sd, (len(teams), n_sims))
        hi = np.array([pos[t] for t in home_ids])
        ai = np.array([pos[t] for t in away_ids])
        dr = r[hi] + self.config.home - r[ai]  # (n_jogos, n_sims)
        p_away = expit(self.c1 - self.beta * dr)
        p_not_home = expit(self.c2 - self.beta * dr)
        u = rng.random(dr.shape)
        home_win = u >= p_not_home
        away_win = u < p_away
        return home_win.astype(np.int64), away_win.astype(np.int64)


def _ordered_probs(dr, beta, c1, c2) -> np.ndarray:
    p_away = expit(c1 - beta * dr)
    p_away_or_draw = expit(c2 - beta * dr)
    return np.column_stack([1 - p_away_or_draw, p_away_or_draw - p_away, p_away])


def _sample_from_outcomes(probs: np.ndarray, n_sims: int, rng: np.random.Generator):
    u = rng.random((len(probs), n_sims))
    home_win = u < probs[:, [0]]
    away_win = u >= (probs[:, [0]] + probs[:, [1]])
    return home_win.astype(np.int64), away_win.astype(np.int64)


def fit_elo(
    data: TrainingData,
    teams: tuple[str, ...] | list[str],
    season: int,
    promoted: set[str] | frozenset[str] = frozenset(),
    config: EloConfig = EloConfig(),
) -> EloModel:
    """Processa em ordem todos os jogos de `data` e devolve os ratings no corte."""
    cfg = config
    ratings: dict[str, float] = {}
    current_season = None
    pre_dr, outcomes = [], []

    def start_season(s, participants):
        first_season = not ratings
        mean = np.mean(list(ratings.values())) if ratings else cfg.initial
        for t in list(ratings):
            ratings[t] = mean + (1 - cfg.season_regress) * (ratings[t] - mean)
        for t in participants:  # estreantes (exceto na primeira temporada do histórico)
            ratings.setdefault(t, mean + (0.0 if first_season else cfg.promoted_offset))

    df = data.matches.sort_values("match_date")
    by_season = {
        s: set(g["home_team_id"]) | set(g["away_team_id"]) for s, g in df.groupby("season")
    }
    for row in df.itertuples(index=False):
        if row.season != current_season:
            current_season = row.season
            start_season(row.season, by_season[row.season])
        rh, ra = ratings[row.home_team_id], ratings[row.away_team_id]
        dr = rh + cfg.home - ra
        expected = 1 / (1 + 10 ** (-dr / 400))
        gd = row.home_goals - row.away_goals
        score = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        delta = cfg.k * _gd_mult(gd) * (score - expected)
        ratings[row.home_team_id] = rh + delta
        ratings[row.away_team_id] = ra - delta
        pre_dr.append(dr)
        outcomes.append(0 if gd > 0 else 1 if gd == 0 else 2)

    if current_season != season:
        # corte antes do primeiro jogo da temporada simulada: aplica a virada de temporada
        start_season(season, set(teams))
    mean = np.mean(list(ratings.values())) if ratings else cfg.initial
    for t in teams:
        ratings.setdefault(t, mean + (cfg.promoted_offset if t in promoted else 0.0))

    beta, c1, c2 = _fit_ordered_logit(np.array(pre_dr), np.array(outcomes))
    return EloModel(dict(ratings), beta, c1, c2, cfg)


def _fit_ordered_logit(dr: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(dr) < 30:  # sem dados suficientes: valores típicos do futebol
        return 0.004, -0.7, 0.4

    def nll(p):
        beta, c1, gap = p
        probs = _ordered_probs(dr, beta, c1, c1 + np.exp(gap))
        return -np.sum(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)))

    res = minimize(nll, x0=[0.004, -0.7, np.log(1.1)], method="Nelder-Mead")
    beta, c1, gap = res.x
    return float(beta), float(c1), float(c1 + np.exp(gap))


@dataclass(frozen=True)
class TableModel:
    """Baseline: taxas de V/E/D de cada time na temporada até o corte, suavizadas."""

    rates: dict[str, np.ndarray]  # time -> [vitória, empate, derrota]
    league: np.ndarray            # [mandante vence, empate, visitante vence] histórico

    def outcome_probs(self, home_ids, away_ids) -> np.ndarray:
        rows = []
        for h, a in zip(home_ids, away_ids):
            rh, ra = self.rates[h], self.rates[a]
            p = np.array([(rh[0] + ra[2]) / 2, (rh[1] + ra[1]) / 2, (rh[2] + ra[0]) / 2])
            # devolve a vantagem de mando que a média simétrica apaga
            p = 0.5 * p / p.sum() + 0.5 * self.league
            rows.append(p / p.sum())
        return np.array(rows)

    def sample_scores(self, home_ids, away_ids, n_sims: int, rng: np.random.Generator):
        return _sample_from_outcomes(self.outcome_probs(home_ids, away_ids), n_sims, rng)


def fit_table(
    data: TrainingData, teams: tuple[str, ...] | list[str], season: int, prior_games: float = 5.0
) -> TableModel:
    df = data.matches
    hist = df[df["season"] < season]
    if len(hist):
        gd = hist["home_goals"] - hist["away_goals"]
        league = np.array([(gd > 0).mean(), (gd == 0).mean(), (gd < 0).mean()])
    else:
        league = np.array([0.48, 0.27, 0.25])
    neutral = np.array([(league[0] + league[2]) / 2, league[1], (league[0] + league[2]) / 2])

    cur = df[df["season"] == season]
    rates = {}
    for t in teams:
        home = cur[cur["home_team_id"] == t]
        away = cur[cur["away_team_id"] == t]
        gf = np.concatenate([home["home_goals"], away["away_goals"]])
        ga = np.concatenate([home["away_goals"], away["home_goals"]])
        counts = np.array([(gf > ga).sum(), (gf == ga).sum(), (gf < ga).sum()], dtype=float)
        rates[t] = (counts + prior_games * neutral) / (counts.sum() + prior_games)
    return TableModel(rates, league)
