-- Uma linha por jogo, vinda do BRA.csv do football-data.co.uk (via Parquet da ingestão).
-- match_date é a data LOCAL (America/Sao_Paulo); kickoff_utc é o horário original (UTC).
-- Odds são de fechamento. Nomes de times ainda crus: padronização em int_matches_unified.
select
    season,
    match_date,
    kickoff_utc,
    home_team,
    away_team,
    home_goals,
    away_goals,
    result,
    is_played,
    odds_psch, odds_pscd, odds_psca,
    odds_maxch, odds_maxcd, odds_maxca,
    odds_avgch, odds_avgcd, odds_avgca,
    odds_bfech, odds_bfecd, odds_bfeca,
    odds_b365ch, odds_b365cd, odds_b365ca
from read_parquet('{{ var("raw_dir") }}/fduk_bra.parquet')
