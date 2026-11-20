"""O site é função pura dos resultados versionados: roda sem dados brutos e não inventa números."""

import json
from pathlib import Path

import pandas as pd
import pytest

from brasileirao_sim.site import build_site, load_results, match_metrics, title_story

RESULTS = Path("results/backtest")
pytestmark = pytest.mark.skipif(not (RESULTS / "match_preds.parquet").exists(),
                                reason="resultados do backtest não versionados")


def test_site_gera_html_e_json(tmp_path):
    index = build_site(tmp_path)
    html = index.read_text(encoding="utf-8")
    assert "<svg" in html and 'id="data"' in html and "http://" not in html.split("<footer>")[0]
    summary = json.loads((tmp_path / "data.json").read_text(encoding="utf-8"))
    assert summary["temporada"] == 2026 and summary["trocas_de_favorito"] >= 1


def test_metricas_do_site_batem_com_recalculo():
    _, games, *_ = load_results()
    m = match_metrics(games).set_index(["model", "periodo"])
    g = games[(games.model == "tabela") & (games.season == 2026)]
    import numpy as np
    p = g[["p_home", "p_draw", "p_away"]].to_numpy()
    idx = g["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    expected = float(-np.log(np.clip(p[np.arange(len(g)), idx], 1e-4, 1)).mean())
    assert m.loc[("tabela", "holdout"), "log_loss"] == pytest.approx(expected)


def test_historia_usa_so_cortes_existentes():
    teams, *_ = load_results()
    story = title_story(teams, {}, 2026, ref_round=999)
    assert story["ref_round"] == story["last_round"]
    assert isinstance(pd.Timestamp(story["cutoff_now"]), pd.Timestamp)
