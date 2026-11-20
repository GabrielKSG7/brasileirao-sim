"""Acesso aos dados e proteção contra vazamento de informação futura.

Regra central do backtest: o modelo só pode ver resultados de jogos com data <= corte.

- `TrainingData` é a ÚNICA forma de entregar resultados aos modelos. Ela só pode ser
  criada por `as_of`, que filtra pela data de corte e verifica o resultado.
- `SeasonState` descreve uma temporada no momento do corte: os jogos já disputados
  (com placar) e os confrontos que faltam (só mandante x visitante, SEM placar).
  Os confrontos restantes são calculados como "todos os pares da temporada menos os já
  disputados". Isso não usa nenhum dado futuro: em pontos corridos, a lista de
  participantes e os confrontos são conhecidos antes do campeonato começar.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MATCH_COLUMNS = [
    "season",
    "match_date",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "is_played",
]


class LeakageError(AssertionError):
    """Algum dado posterior à data de corte chegou ao modelo."""


def load_matches(db_path: str | Path = "data/brasileirao.duckdb") -> pd.DataFrame:
    """Lê o mart fct_matches gerado pelo dbt."""
    import duckdb

    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("select * from main.fct_matches").df()
    return normalize_matches(df)


def normalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Garante tipos consistentes (data como datetime.date, gols inteiros anuláveis)."""
    missing = [c for c in MATCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"colunas ausentes: {missing}")
    out = df.copy()
    out["match_date"] = pd.to_datetime(out["match_date"]).dt.date
    out["season"] = out["season"].astype("int64")
    out["home_goals"] = out["home_goals"].astype("Int64")
    out["away_goals"] = out["away_goals"].astype("Int64")
    out["is_played"] = out["is_played"].astype(bool)
    return out.sort_values(["match_date", "home_team_id"]).reset_index(drop=True)


@dataclass(frozen=True)
class TrainingData:
    """Jogos disputados até `cutoff` (inclusive). Criar sempre via `as_of`."""

    matches: pd.DataFrame
    cutoff: dt.date

    def __post_init__(self) -> None:
        if len(self.matches) and max(self.matches["match_date"]) > self.cutoff:
            raise LeakageError(
                f"jogo de {max(self.matches['match_date'])} em dados com corte {self.cutoff}"
            )
        if len(self.matches) and not self.matches["is_played"].all():
            raise LeakageError("TrainingData só aceita jogos disputados")


def as_of(matches: pd.DataFrame, cutoff: dt.date, lookback_days: int | None = None) -> TrainingData:
    """Resultados conhecidos na data de corte (opcionalmente só os últimos `lookback_days`)."""
    keep = matches["is_played"] & (matches["match_date"] <= cutoff)
    if lookback_days is not None:
        keep &= matches["match_date"] > cutoff - dt.timedelta(days=lookback_days)
    sub = matches.loc[keep, MATCH_COLUMNS].copy()
    sub["home_goals"] = sub["home_goals"].astype("int64")
    sub["away_goals"] = sub["away_goals"].astype("int64")
    return TrainingData(matches=sub.reset_index(drop=True), cutoff=cutoff)


@dataclass(frozen=True)
class SeasonState:
    season: int
    cutoff: dt.date
    teams: tuple[str, ...]
    played: pd.DataFrame      # jogos da temporada até o corte, com placar
    remaining: pd.DataFrame   # só home_team_id, away_team_id (sem placar)
    promoted: frozenset[str]  # times que não estavam na Série A na temporada anterior

    @property
    def round_estimate(self) -> float:
        """Rodadas disputadas, em média (jogos disputados / 10)."""
        return len(self.played) / (len(self.teams) / 2)


def season_teams(matches: pd.DataFrame, season: int) -> tuple[str, ...]:
    s = matches[matches["season"] == season]
    teams = sorted(set(s["home_team_id"]) | set(s["away_team_id"]))
    if not teams:
        raise ValueError(f"temporada {season} sem jogos")
    return tuple(teams)


def season_state(matches: pd.DataFrame, season: int, cutoff: dt.date) -> SeasonState:
    teams = season_teams(matches, season)
    known = as_of(matches, cutoff).matches
    played = known[known["season"] == season].reset_index(drop=True)

    done = set(zip(played["home_team_id"], played["away_team_id"]))
    remaining = pd.DataFrame(
        [(h, a) for h in teams for a in teams if h != a and (h, a) not in done],
        columns=["home_team_id", "away_team_id"],
    )
    try:
        previous = set(season_teams(matches, season - 1))
    except ValueError:
        previous = set(teams)  # sem temporada anterior: ninguém é tratado como promovido
    return SeasonState(
        season=season,
        cutoff=cutoff,
        teams=teams,
        played=played,
        remaining=remaining,
        promoted=frozenset(t for t in teams if t not in previous),
    )


def preseason_cutoff(matches: pd.DataFrame, season: int) -> dt.date:
    """Véspera do primeiro jogo da temporada."""
    first = matches.loc[matches["season"] == season, "match_date"].min()
    return first - dt.timedelta(days=1)


def round_cutoffs(matches: pd.DataFrame, season: int) -> list[tuple[int, dt.date]]:
    """Datas de corte por rodada: (0, véspera da temporada), (k, data em que o total de jogos
    disputados atinge 10*k). Usa a quantidade de jogos, não o número oficial da rodada, porque
    jogos adiados embaralham a ordem das rodadas."""
    s = matches[(matches["season"] == season) & matches["is_played"]]
    per_round = len(season_teams(matches, season)) // 2
    dates = sorted(s["match_date"])
    cutoffs = [(0, preseason_cutoff(matches, season))]
    k = 1
    while k * per_round <= len(dates):
        cutoffs.append((k, dates[k * per_round - 1]))
        k += 1
    return cutoffs
