import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from brasileirao_sim.ingest import footballdata_uk as fd

FIXTURE = Path(__file__).parent / "fixtures" / "BRA_synthetic.csv"


def test_parse_types_and_unplayed_match():
    df = fd.parse_bra_csv(FIXTURE)
    assert len(df) == 5
    assert df["home_goals"].dtype == "Int64"
    # o último jogo ainda não aconteceu: sem placar e sem resultado
    assert int((~df["is_played"]).sum()) == 1


def test_local_date_converts_from_utc():
    df = fd.parse_bra_csv(FIXTURE).set_index(["home_team", "away_team"])
    # 02/02 01:00 UTC = 01/02 22:00 em Brasília
    assert df.loc[("Flamengo RJ", "Palmeiras"), "match_date"] == dt.date(2099, 2, 1)
    # 01/02 21:30 UTC = 01/02 18:30 em Brasília (mesmo dia)
    assert df.loc[("Sao Paulo", "Corinthians"), "match_date"] == dt.date(2099, 2, 1)
    # 08/02 23:00 UTC = 08/02 20:00 em Brasília
    assert df.loc[("Corinthians", "Flamengo RJ"), "match_date"] == dt.date(2099, 2, 8)


def test_odds_columns_are_closing_odds():
    df = fd.parse_bra_csv(FIXTURE)
    assert "odds_avgch" in df.columns and "odds_psch" in df.columns
    assert df["odds_avgch"].notna().all()
    assert df["odds_psch"].isna().sum() == 3  # Pinnacle ausente em 3 jogos


def test_missing_required_column_fails(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.read_csv(FIXTURE).drop(columns=["HG"]).to_csv(bad, index=False)
    with pytest.raises(fd.SchemaError, match="HG"):
        fd.parse_bra_csv(bad)


def test_result_must_match_score(tmp_path):
    bad = tmp_path / "bad.csv"
    df = pd.read_csv(FIXTURE)
    df.loc[0, "Res"] = "A"  # 2x1 não pode ser vitória do visitante
    df.to_csv(bad, index=False)
    with pytest.raises(fd.SchemaError, match="não bate"):
        fd.parse_bra_csv(bad)


def test_same_pairing_twice_in_a_season_fails(tmp_path):
    bad = tmp_path / "bad.csv"
    df = pd.read_csv(FIXTURE)
    dup = df.iloc[[0]].copy()
    dup["Date"] = "20/02/2099"  # mesmo mandante x visitante, outra data
    pd.concat([df, dup]).to_csv(bad, index=False)
    with pytest.raises(fd.SchemaError, match="duplicados"):
        fd.parse_bra_csv(bad)


def test_run_writes_parquet_and_manifest(tmp_path):
    out = fd.run(FIXTURE, tmp_path)
    assert out.exists()
    manifest = (tmp_path / "fduk_bra.manifest.json").read_text(encoding="utf-8")
    assert '"played_rows": 4' in manifest
