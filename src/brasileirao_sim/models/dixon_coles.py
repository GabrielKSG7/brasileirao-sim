"""Modelo Dixon-Coles (1997) com decaimento temporal.

Gols do mandante ~ Poisson(lambda), gols do visitante ~ Poisson(mu), com
    log(lambda) = base + mando + ataque[mandante]  - defesa[visitante]
    log(mu)     = base +         ataque[visitante] - defesa[mandante]
e a correção tau(rho) de Dixon-Coles para placares baixos (0x0, 1x0, 0x1, 1x1).

Escolhas para estabilidade com poucos jogos:
- peso de cada jogo = exp(-xi * dias até o corte) (jogos antigos pesam menos);
- penalização ridge que puxa ataque/defesa para uma média a priori: 0 para quem já
  estava na Série A e valores negativos para promovidos (mais fracos, em média);
- ajuste em duas etapas: parâmetros de Poisson por L-BFGS com gradiente analítico,
  depois rho por busca unidimensional com os demais fixos.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from brasileirao_sim.data import TrainingData


@dataclass(frozen=True)
class DCConfig:
    xi: float = 0.0019              # decaimento por dia (meia-vida ~ 1 ano)
    ridge: float = 5.0              # força da penalização para a média a priori
    promoted_attack: float = -0.15  # média a priori do ataque de promovidos
    promoted_defense: float = -0.15  # média a priori da defesa de promovidos
    lookback_days: int = 4 * 365    # jogos mais antigos que isso são ignorados
    max_goals: int = 10             # placar máximo considerado por time
    # Incerteza de parâmetro na simulação: desvio-padrão do ruído somado ao ataque e à defesa
    # de cada time, sorteado uma vez por simulação (vale para o restante da temporada).
    sim_strength_sd: float = 0.0


@dataclass(frozen=True)
class DCModel:
    teams: tuple[str, ...]
    attack: np.ndarray
    defense: np.ndarray
    base: float
    home: float
    rho: float
    config: DCConfig = field(repr=False)

    def _idx(self, team_ids) -> np.ndarray:
        pos = {t: i for i, t in enumerate(self.teams)}
        try:
            return np.array([pos[t] for t in team_ids], dtype=int)
        except KeyError as exc:  # time sem parâmetro: erro de uso, não silenciar
            raise KeyError(f"time sem parâmetros no modelo: {exc}") from None

    def rates(self, home_ids, away_ids) -> tuple[np.ndarray, np.ndarray]:
        h, a = self._idx(home_ids), self._idx(away_ids)
        lam = np.exp(self.base + self.home + self.attack[h] - self.defense[a])
        mu = np.exp(self.base + self.attack[a] - self.defense[h])
        return lam, mu

    def score_matrices(self, home_ids, away_ids) -> np.ndarray:
        """Probabilidade de cada placar, shape (n_jogos, G+1, G+1), normalizada."""
        lam, mu = self.rates(home_ids, away_ids)
        g = np.arange(self.config.max_goals + 1)
        ph = poisson.pmf(g[None, :], lam[:, None])
        pa = poisson.pmf(g[None, :], mu[:, None])
        m = ph[:, :, None] * pa[:, None, :]
        r = self.rho
        m[:, 0, 0] *= 1 - lam * mu * r
        m[:, 0, 1] *= 1 + lam * r
        m[:, 1, 0] *= 1 + mu * r
        m[:, 1, 1] *= 1 - r
        m = np.clip(m, 0, None)
        return m / m.sum(axis=(1, 2), keepdims=True)

    def outcome_probs(self, home_ids, away_ids) -> np.ndarray:
        """(n_jogos, 3): probabilidade de vitória do mandante, empate, vitória do visitante."""
        m = self.score_matrices(home_ids, away_ids)
        home = np.tril(np.ones(m.shape[1:]), -1)
        p_home = (m * home).sum(axis=(1, 2))
        p_draw = np.trace(m, axis1=1, axis2=2)
        return np.column_stack([p_home, p_draw, 1 - p_home - p_draw])

    def sample_scores(self, home_ids, away_ids, n_sims: int, rng: np.random.Generator):
        """Sorteia placares: dois arrays (n_jogos, n_sims).

        Amostragem exata de Dixon-Coles por aceitação/rejeição: propõe placares de Poissons
        independentes e aceita com probabilidade tau/max(tau). Com `sim_strength_sd` > 0, cada
        simulação usa forças perturbadas (incerteza sobre o nível real de cada time)."""
        h, a = self._idx(home_ids), self._idx(away_ids)
        n_fix = len(h)
        att = np.repeat(self.attack[:, None], n_sims, axis=1)
        dfn = np.repeat(self.defense[:, None], n_sims, axis=1)
        sd = self.config.sim_strength_sd
        if sd > 0:
            att = att + rng.normal(0, sd, att.shape)
            dfn = dfn + rng.normal(0, sd, dfn.shape)
        lam = np.exp(self.base + self.home + att[h] - dfn[a])
        mu = np.exp(self.base + att[a] - dfn[h])

        r = self.rho
        tau_max = np.maximum.reduce(
            [np.ones_like(lam), 1 - lam * mu * r, 1 + lam * r, 1 + mu * r, np.full_like(lam, 1 - r)]
        )
        hg = np.empty((n_fix, n_sims), dtype=np.int64)
        ag = np.empty((n_fix, n_sims), dtype=np.int64)
        todo = np.ones((n_fix, n_sims), dtype=bool)
        for _ in range(50):
            if not todo.any():
                break
            x = rng.poisson(lam[todo])
            y = rng.poisson(mu[todo])
            t = _tau(x, y, lam[todo], mu[todo], r)
            ok = rng.random(x.shape) * tau_max[todo] < t
            idx = np.flatnonzero(todo)[ok]
            hg.flat[idx] = x[ok]
            ag.flat[idx] = y[ok]
            todo.flat[idx] = False
        if todo.any():  # praticamente impossível; mantém a proposta independente
            hg[todo] = rng.poisson(lam[todo])
            ag[todo] = rng.poisson(mu[todo])
        return hg, ag


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def fit(
    data: TrainingData,
    teams: tuple[str, ...] | list[str],
    promoted: set[str] | frozenset[str] = frozenset(),
    config: DCConfig = DCConfig(),
) -> DCModel:
    """Ajusta o modelo com os jogos de `data` (até o corte). `teams` são os times que precisam
    de parâmetros (os da temporada simulada); times que só aparecem no histórico também entram.
    Um time sem nenhum jogo no histórico fica com a média a priori."""
    df = data.matches
    if config.lookback_days is not None:
        df = df[df["match_date"] > data.cutoff - dt.timedelta(days=config.lookback_days)]

    all_teams = tuple(sorted(set(teams) | set(df["home_team_id"]) | set(df["away_team_id"])))
    pos = {t: i for i, t in enumerate(all_teams)}
    n = len(all_teams)
    prior_a = np.array([config.promoted_attack if t in promoted else 0.0 for t in all_teams])
    prior_d = np.array([config.promoted_defense if t in promoted else 0.0 for t in all_teams])

    if len(df) == 0:
        return DCModel(all_teams, prior_a, prior_d, np.log(1.3), 0.25, 0.0, config)

    h = df["home_team_id"].map(pos).to_numpy()
    a = df["away_team_id"].map(pos).to_numpy()
    x = df["home_goals"].to_numpy(dtype=float)
    y = df["away_goals"].to_numpy(dtype=float)
    age = np.array([(data.cutoff - d).days for d in df["match_date"]], dtype=float)
    w = np.exp(-config.xi * age)

    def objective(theta):
        base, home = theta[0], theta[1]
        att, dfn = theta[2 : 2 + n], theta[2 + n :]
        eta1 = base + home + att[h] - dfn[a]
        eta2 = base + att[a] - dfn[h]
        l1, l2 = np.exp(eta1), np.exp(eta2)
        ll = np.sum(w * (x * eta1 - l1 + y * eta2 - l2))
        pen = config.ridge * (np.sum((att - prior_a) ** 2) + np.sum((dfn - prior_d) ** 2))
        r1, r2 = w * (x - l1), w * (y - l2)
        g = np.empty_like(theta)
        g[0] = r1.sum() + r2.sum()
        g[1] = r1.sum()
        g[2 : 2 + n] = np.bincount(h, r1, n) + np.bincount(a, r2, n)
        g[2 + n :] = -np.bincount(a, r1, n) - np.bincount(h, r2, n)
        g = -g
        g[2 : 2 + n] += 2 * config.ridge * (att - prior_a)
        g[2 + n :] += 2 * config.ridge * (dfn - prior_d)
        return -ll + pen, g

    theta0 = np.concatenate([[np.log(max(x.mean(), 0.1)), 0.2], prior_a, prior_d])
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Dixon-Coles não convergiu: {res.message}")
    th = res.x
    base, home, att, dfn = th[0], th[1], th[2 : 2 + n], th[2 + n :]

    lam = np.exp(base + home + att[h] - dfn[a])
    mu = np.exp(base + att[a] - dfn[h])
    # limites que mantêm tau > 0 em todos os jogos
    hi = min(1.0 / np.max(lam * mu), 0.99)
    lo = max(-1.0 / np.max(lam), -1.0 / np.max(mu), -0.99)

    def neg_ll_rho(rho):
        return -np.sum(w * np.log(np.clip(_tau(x, y, lam, mu, rho), 1e-12, None)))

    rho = minimize_scalar(neg_ll_rho, bounds=(max(lo, -0.3), min(hi, 0.3)), method="bounded").x
    return DCModel(all_teams, att, dfn, float(base), float(home), float(rho), config)
