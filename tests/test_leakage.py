"""O modelo nunca pode enxergar resultados posteriores à data de corte."""

import datetime as dt

import numpy as np
import pytest

from brasileirao_sim.data import LeakageError, TrainingData, as_of, round_cutoffs, season_state
from brasileirao_sim.simulate import forecast

from synthetic import make_league

MATCHES, _, _ = make_league()
SEASON = 2098
CUTOFF = dict(round_cutoffs(MATCHES, SEASON))[9]


def test_as_of_drops_matches_after_cutoff():
    data = as_of(MATCHES, CUTOFF)
    assert max(data.matches["match_date"]) <= CUTOFF
    assert len(data.matches) < len(MATCHES)


def test_training_data_rejects_future_rows():
    future = MATCHES[MATCHES["match_date"] > CUTOFF]
    with pytest.raises(LeakageError):
        TrainingData(matches=future, cutoff=CUTOFF)


def test_remaining_fixtures_carry_no_scores():
    state = season_state(MATCHES, SEASON, CUTOFF)
    assert list(state.remaining.columns) == ["home_team_id", "away_team_id"]
    # jogados + restantes = todos os confrontos da temporada, sem sobreposição
    n = len(state.teams)
    assert len(state.played) + len(state.remaining) == n * (n - 1)
    played = set(zip(state.played["home_team_id"], state.played["away_team_id"]))
    assert played.isdisjoint(zip(state.remaining["home_team_id"], state.remaining["away_team_id"]))


@pytest.mark.parametrize("model", ["dixon_coles", "elo", "tabela"])
def test_future_results_do_not_change_forecast(model):
    """Teste de perturbação: inverter todos os placares depois do corte não pode mudar nada."""
    tampered = MATCHES.copy()
    future = tampered["match_date"] > CUTOFF
    tampered.loc[future, ["home_goals", "away_goals"]] = tampered.loc[
        future, ["away_goals", "home_goals"]
    ].to_numpy() * 3

    a, _, _ = forecast(MATCHES, SEASON, CUTOFF, model, n_sims=2000, seed=1)
    b, _, _ = forecast(tampered, SEASON, CUTOFF, model, n_sims=2000, seed=1)
    np.testing.assert_array_equal(a.position_probs, b.position_probs)


def test_round_cutoffs_are_increasing_and_start_before_season():
    cuts = round_cutoffs(MATCHES, SEASON)
    dates = [d for _, d in cuts]
    assert dates == sorted(dates)
    assert dates[0] < min(MATCHES.loc[MATCHES["season"] == SEASON, "match_date"])
    assert cuts[0][0] == 0 and cuts[-1][0] == 18


def test_preseason_state_has_no_played_matches():
    cut = round_cutoffs(MATCHES, SEASON)[0][1]
    state = season_state(MATCHES, SEASON, cut)
    assert len(state.played) == 0
    assert cut < dt.date(SEASON, 2, 1)
