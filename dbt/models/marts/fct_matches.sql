-- Fato de jogos pronto para a modelagem em Python.
-- market_p_*: probabilidades implícitas na odd média de fechamento, sem a margem da casa
-- (normalização proporcional). Servem só como BENCHMARK, nunca como entrada do modelo.
with m as (
    select * from {{ ref('int_matches_unified') }}
),
implied as (
    select
        *,
        1.0 / odds_avgch as ih,
        1.0 / odds_avgcd as id,
        1.0 / odds_avgca as ia
    from m
)
select
    season,
    match_date,
    kickoff_utc,
    home_team_id,
    away_team_id,
    home_goals,
    away_goals,
    result,
    is_played,
    ih / (ih + id + ia) as market_p_home,
    id / (ih + id + ia) as market_p_draw,
    ia / (ih + id + ia) as market_p_away
from implied
