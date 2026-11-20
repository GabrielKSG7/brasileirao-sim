import datetime as dt

import numpy as np
from scipy.stats import spearmanr

from brasileirao_sim.data import as_of
from brasileirao_sim.models import dixon_coles, elo

from synthetic import make_league

MATCHES, ATTACK, DEFENSE = make_league(n_teams=12, seasons=(2095, 2096, 2097, 2098))
CUT = dt.date(2099, 1, 1)
TEAMS = tuple(sorted(ATTACK))


def _dc():
    return dixon_coles.fit(as_of(MATCHES, CUT), TEAMS, config=dixon_coles.DCConfig(ridge=1.0))


def test_dc_recovers_team_strengths_and_home_advantage():
    m = _dc()
    att = [m.attack[m.teams.index(t)] for t in TEAMS]
    dfn = [m.defense[m.teams.index(t)] for t in TEAMS]
    assert spearmanr(att, [ATTACK[t] for t in TEAMS]).statistic > 0.85
    assert spearmanr(dfn, [DEFENSE[t] for t in TEAMS]).statistic > 0.85
    assert 0.15 < m.home < 0.45  # valor real: 0.3


def test_dc_probabilities_are_valid():
    m = _dc()
    mats = m.score_matrices(["t00", "t11"], ["t11", "t00"])
    np.testing.assert_allclose(mats.sum(axis=(1, 2)), 1.0)
    probs = m.outcome_probs(["t00", "t11"], ["t11", "t00"])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] > 0.6  # o mais forte em casa contra o mais fraco
    assert probs[1, 2] > probs[1, 0]  # o mais fraco em casa ainda é azarão


def test_dc_sampling_matches_probabilities():
    m = _dc()
    rng = np.random.default_rng(0)
    hg, ag = m.sample_scores(["t03"], ["t08"], 40_000, rng)
    freq = np.array([(hg > ag).mean(), (hg == ag).mean(), (hg < ag).mean()])
    np.testing.assert_allclose(freq, m.outcome_probs(["t03"], ["t08"])[0], atol=0.01)


def test_dc_team_without_history_gets_prior():
    cfg = dixon_coles.DCConfig()
    m = dixon_coles.fit(as_of(MATCHES, CUT), (*TEAMS, "novato"), {"novato"}, cfg)
    i = m.teams.index("novato")
    assert m.attack[i] == cfg.promoted_attack and m.defense[i] == cfg.promoted_defense


def test_dc_decay_weights_recent_matches_more():
    """Com decaimento forte, só a temporada mais recente importa: um time que mudou de
    nível deve refletir o nível novo."""
    df = MATCHES.copy()
    last = df["season"] == 2098
    df.loc[last & (df["home_team_id"] == "t11"), "home_goals"] += 3
    df.loc[last & (df["away_team_id"] == "t11"), "away_goals"] += 3
    slow = dixon_coles.fit(as_of(df, CUT), TEAMS, config=dixon_coles.DCConfig(xi=0.0, ridge=1))
    fast = dixon_coles.fit(as_of(df, CUT), TEAMS, config=dixon_coles.DCConfig(xi=0.01, ridge=1))
    assert fast.attack[fast.teams.index("t11")] > slow.attack[slow.teams.index("t11")]


def test_elo_ranks_teams_and_probabilities_sum_to_one():
    m = elo.fit_elo(as_of(MATCHES, CUT), TEAMS, season=2099)
    ratings = [m.ratings[t] for t in TEAMS]
    assert spearmanr(ratings, [ATTACK[t] + DEFENSE[t] for t in TEAMS]).statistic > 0.85
    probs = m.outcome_probs(["t00", "t05"], ["t11", "t06"])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert (probs > 0).all()


def test_elo_updates_are_zero_sum_within_a_season():
    one_season = MATCHES[MATCHES["season"] == 2095]
    m = elo.fit_elo(as_of(one_season, CUT), TEAMS, season=2095)
    assert abs(np.mean(list(m.ratings.values())) - 1500) < 1e-6


def test_table_baseline_uses_only_current_season():
    m = elo.fit_table(as_of(MATCHES, CUT), TEAMS, season=2099)  # temporada sem jogos
    first = m.rates[TEAMS[0]]
    assert all(np.allclose(m.rates[t], first) for t in TEAMS)  # todos iguais: só a priori
    np.testing.assert_allclose(m.outcome_probs(["t00"], ["t01"]).sum(), 1.0)
