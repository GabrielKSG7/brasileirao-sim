-- Uma linha por jogo, vinda do BRA.csv do football-data.co.uk (via Parquet da ingestão).
-- Nomes de times ainda crus: a padronização acontece em int_matches_unified.
select
    season,
    match_date,
    kickoff_time,
    home_team,
    away_team,
    home_goals,
    away_goals,
    result,
    home_goals is not null and away_goals is not null as is_played,
    odds_ph, odds_pd, odds_pa,
    odds_maxh, odds_maxd, odds_maxa,
    odds_avgh, odds_avgd, odds_avga
from read_parquet('{{ var("raw_dir") }}/fduk_bra.parquet')
