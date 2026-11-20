import numpy as np

from brasileirao_sim.data import round_cutoffs, season_state
from brasileirao_sim.simulate import ZONES, forecast, rank

from synthetic import make_league

MATCHES, _, _ = make_league(n_teams=20, seasons=(2097, 2098))


def test_rank_tiebreakers_points_then_wins_then_goal_difference():
    rng = np.random.default_rng(0)
    pts = np.array([[10, 10, 10, 12]], dtype=float)
    wins = np.array([[3, 2, 2, 3]], dtype=float)
    gf = np.array([[5, 9, 6, 1]], dtype=float)
    ga = np.array([[5, 4, 4, 9]], dtype=float)
    # time 3 lidera nos pontos; entre 0,1,2 empatados em pontos, 0 tem mais vitórias;
    # 1 e 2 empatam em vitórias e 1 tem saldo maior
    assert rank(pts, wins, gf, ga, rng).tolist() == [[1, 2, 3, 0]]


def test_finished_season_reproduces_the_real_table():
    last_cut = round_cutoffs(MATCHES, 2098)[-1][1]
    fc, _, state = forecast(MATCHES, 2098, last_cut, "dixon_coles", n_sims=200)
    assert len(state.remaining) == 0
    assert set(np.unique(fc.position_probs)) <= {0.0, 1.0}  # sem incerteza
    np.testing.assert_allclose(fc.expected_points, fc.current_points)


def test_probabilities_are_consistent():
    cut = dict(round_cutoffs(MATCHES, 2098))[10]
    fc, _, _ = forecast(MATCHES, 2098, cut, "dixon_coles", n_sims=3000)
    np.testing.assert_allclose(fc.position_probs.sum(axis=0), 1.0)
    np.testing.assert_allclose(fc.position_probs.sum(axis=1), 1.0)
    z = fc.zone_probs()
    # soma de cada zona = número de vagas
    for zone, (lo, hi) in ZONES.items():
        assert abs(z[zone].sum() - (hi - lo + 1)) < 1e-9
    assert (z["g4"] >= z["titulo"]).all()


def test_same_seed_same_result():
    cut = dict(round_cutoffs(MATCHES, 2098))[5]
    a, _, _ = forecast(MATCHES, 2098, cut, "elo", n_sims=1000, seed=3)
    b, _, _ = forecast(MATCHES, 2098, cut, "elo", n_sims=1000, seed=3)
    np.testing.assert_array_equal(a.position_probs, b.position_probs)


def test_current_points_match_played_games():
    cut = dict(round_cutoffs(MATCHES, 2098))[7]
    state = season_state(MATCHES, 2098, cut)
    fc, _, _ = forecast(MATCHES, 2098, cut, "tabela", n_sims=100)
    p = state.played
    hp = np.where(p.home_goals > p.away_goals, 3, np.where(p.home_goals == p.away_goals, 1, 0))
    ap = np.where(p.home_goals < p.away_goals, 3, np.where(p.home_goals == p.away_goals, 1, 0))
    total = dict.fromkeys(state.teams, 0)
    for h, a, x, y in zip(p.home_team_id, p.away_team_id, hp, ap):
        total[h] += x
        total[a] += y
    assert fc.current_points.tolist() == [total[t] for t in state.teams]
