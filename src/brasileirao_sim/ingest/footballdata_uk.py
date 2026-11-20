"""Ingestão do BRA.csv do football-data.co.uk (resultados + odds do Brasileirão).

O arquivo é baixado como veio (data/raw/BRA.csv), validado e convertido em Parquet
com tipos corretos. Um manifesto JSON guarda o hash e a data da coleta, para que
cada rodada do backtest possa citar exatamente os dados que usou.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.football-data.co.uk/new/BRA.csv"
USER_AGENT = "brasileirao-sim (+https://github.com/GabrielKSG7/brasileirao-sim)"

# Colunas indispensáveis: sem elas a ingestão falha, em vez de seguir com dado errado.
REQUIRED = ["Season", "Date", "Home", "Away", "HG", "AG", "Res"]
# Colunas opcionais (odds): se faltarem, viram vazias e um aviso é emitido.
OPTIONAL = ["Country", "League", "Time", "PH", "PD", "PA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"]
ODDS = ["PH", "PD", "PA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"]


class SchemaError(ValueError):
    """O arquivo não tem o formato esperado."""


def parse_bra_csv(path: str | Path) -> pd.DataFrame:
    """Lê e normaliza o CSV. Jogos sem placar (ainda não disputados) ficam com gols nulos."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    raw.columns = [c.strip() for c in raw.columns]

    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise SchemaError(
            f"Colunas obrigatórias ausentes: {missing}. Colunas encontradas: {list(raw.columns)}"
        )
    for col in OPTIONAL:
        if col not in raw.columns:
            print(f"aviso: coluna opcional ausente: {col}", file=sys.stderr)
            raw[col] = pd.NA

    df = pd.DataFrame()
    df["season"] = pd.to_numeric(raw["Season"], errors="raise").astype("int64")
    df["match_date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="raise").dt.date
    df["kickoff_time"] = raw["Time"].astype("string")
    df["home_team"] = raw["Home"].str.strip()
    df["away_team"] = raw["Away"].str.strip()
    df["home_goals"] = pd.to_numeric(raw["HG"], errors="raise").astype("Int64")
    df["away_goals"] = pd.to_numeric(raw["AG"], errors="raise").astype("Int64")
    df["result"] = raw["Res"].str.strip().astype("string")
    for col in ODDS:
        df[f"odds_{col.lower()}"] = pd.to_numeric(raw[col], errors="coerce")

    _validate(df)
    return df.sort_values(["match_date", "home_team"]).reset_index(drop=True)


def _validate(df: pd.DataFrame) -> None:
    played = df.dropna(subset=["home_goals", "away_goals"])
    bad_res = played[~played["result"].isin(["H", "D", "A"])]
    if len(bad_res):
        raise SchemaError(f"{len(bad_res)} jogos com resultado fora de H/D/A")

    expected = pd.Series("D", index=played.index)
    expected[played["home_goals"] > played["away_goals"]] = "H"
    expected[played["home_goals"] < played["away_goals"]] = "A"
    mismatch = played[expected != played["result"]]
    if len(mismatch):
        raise SchemaError(f"{len(mismatch)} jogos em que Res não bate com o placar")

    if (played[["home_goals", "away_goals"]] < 0).any().any():
        raise SchemaError("gols negativos")
    if (df["home_team"] == df["away_team"]).any():
        raise SchemaError("jogo com o mesmo time nos dois lados")

    dup = df.duplicated(["match_date", "home_team", "away_team"])
    if dup.any():
        raise SchemaError(f"{int(dup.sum())} jogos duplicados (data, mandante, visitante)")


def download(dest: Path, url: str = URL) -> Path:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def run(csv_path: Path, out_dir: Path, source_url: str | None = None) -> Path:
    df = parse_bra_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "fduk_bra.parquet"
    df.to_parquet(parquet, index=False)

    manifest = {
        "source": source_url or "arquivo local",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "rows": len(df),
        "played_rows": int(df["home_goals"].notna().sum()),
        "seasons": sorted(int(s) for s in df["season"].unique()),
        "first_date": str(df["match_date"].min()),
        "last_date": str(df["match_date"].max()),
    }
    (out_dir / "fduk_bra.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return parquet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingere o BRA.csv do football-data.co.uk em Parquet.")
    p.add_argument("--csv", type=Path, help="usa um CSV já baixado em vez de baixar da internet")
    p.add_argument("--out", type=Path, default=Path("data/raw"), help="pasta de saída")
    args = p.parse_args(argv)

    if args.csv:
        csv_path, src = args.csv, None
    else:
        csv_path, src = download(args.out / "BRA.csv"), URL
    run(csv_path, args.out, src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
