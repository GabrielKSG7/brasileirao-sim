import numpy as np
import pandas as pd

from brasileirao_sim.backtest import ModelSpec, final_positions, run_season, season_is_complete
from brasileirao_sim.metrics import binary_scores, calibration_table, match_scores

from synthetic import make_league


def test_rps_known_values():
    res = pd.Series(["H", "H", "A"])
    probs = np.array([[1, 0, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]])
    sc = match_scores(probs, res)
    np.testing.assert_allclose(sc["rps"], [0.0, 1.0, 0.5 * ((1 / 3) ** 2 + (2 / 3) ** 2)])
    # errar "por pouco" (empate) custa menos que errar feio (visitante)
    near = match_scores(np.array([[0, 1, 0]]), pd.Series(["H"]))["rps"][0]
    far = match_scores(np.array([[0, 0, 1]]), pd.Series(["H"]))["rps"][0]
    assert near < far


def test_log_loss_and_brier():
    sc = match_scores(np.array([[0.5, 0.3, 0.2]]), pd.Series(["D"]))
    np.testing.assert_allclose(sc["log_loss"], -np.log(0.3))
    np.testing.assert_allclose(sc["brier"], 0.25 + 0.49 + 0.04)


def test_binary_scores_clip_certainty():
    sc = binary_scores(np.array([0.0, 1.0]), np.array([1, 1]))
    assert np.isfinite(sc["log_loss"]).all()
    np.testing.assert_allclose(sc["brier"], [1.0, 0.0])


def test_calibration_table_perfect_model():
    rng = np.random.default_rng(0)
    p = rng.random(50_000)
    o = rng.random(50_000) < p
    t = calibration_table(p, o)
    np.testing.assert_allclose(t["prob_media"], t["freq_observada"], atol=0.02)


MATCHES, _, _ = make_league(n_teams=20, seasons=(2097, 2098))


def test_backtest_predicts_each_match_once_before_it_happens():
    _, games = run_season(MATCHES, 2098, [ModelSpec("elo")], n_sims=0)
    assert len(games) == 380  # todos os jogos da temporada, uma vez cada
    assert (games["cutoff"] < games["match_date"]).all()
    assert not games.duplicated(["home_team_id", "away_team_id"]).any()


def test_backtest_team_probs_shape():
    teams, _ = run_season(MATCHES, 2098, [ModelSpec("tabela")], n_sims=200, rounds=[0, 10, 38])
    assert sorted(teams["round"].unique()) == [0, 10, 38]
    assert teams.groupby("round")["titulo"].sum().round(6).eq(1).all()
    last = teams[teams["round"] == 38]
    final = final_positions(MATCHES, 2098).set_index("team_id")
    # no último corte a temporada acabou: a probabilidade de título é 1 para o campeão real
    champion = final.index[final["final_position"] == 1][0]
    assert last.set_index("team_id").loc[champion, "titulo"] == 1.0


def test_season_is_complete():
    assert season_is_complete(MATCHES, 2098)
    partial = MATCHES[~((MATCHES["season"] == 2098) & (MATCHES.index % 7 == 0))]
    assert not season_is_complete(partial, 2098)


def test_frozen_params_load():
    from pathlib import Path

    from brasileirao_sim.tune import load_frozen_specs

    path = Path(__file__).resolve().parents[1] / "config" / "model_params.json"
    specs = {s.name: s for s in load_frozen_specs(path)}
    assert set(specs) == {"dixon_coles", "elo", "tabela"}
    assert specs["dixon_coles"].dc.sim_strength_sd > 0
