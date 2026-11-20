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
    assert df["home_goals"].isna().sum() == 1
    assert df["match_date"].min().isoformat() == "2099-02-01"


def test_dates_are_day_first():
    df = fd.parse_bra_csv(FIXTURE)
    assert df["match_date"].max().isoformat() == "2099-02-15"


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


def test_duplicate_match_fails(tmp_path):
    bad = tmp_path / "bad.csv"
    df = pd.read_csv(FIXTURE)
    pd.concat([df, df.iloc[[0]]]).to_csv(bad, index=False)
    with pytest.raises(fd.SchemaError, match="duplicados"):
        fd.parse_bra_csv(bad)


def test_run_writes_parquet_and_manifest(tmp_path):
    out = fd.run(FIXTURE, tmp_path)
    assert out.exists()
    manifest = (tmp_path / "fduk_bra.manifest.json").read_text(encoding="utf-8")
    assert '"played_rows": 4' in manifest
