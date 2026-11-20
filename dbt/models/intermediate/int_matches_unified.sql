-- Jogos com nomes de times padronizados (seed team_aliases).
-- Nome sem correspondência fica com team_id nulo e é pego pelo teste assert_no_unmapped_teams.
select
    m.season,
    m.match_date,
    m.kickoff_time,
    m.home_team as home_team_raw,
    m.away_team as away_team_raw,
    h.team_id as home_team_id,
    a.team_id as away_team_id,
    m.home_goals,
    m.away_goals,
    m.result,
    m.is_played,
    m.odds_ph, m.odds_pd, m.odds_pa,
    m.odds_avgh, m.odds_avgd, m.odds_avga
from {{ ref('stg_fduk_matches') }} as m
left join {{ ref('team_aliases') }} as h on h.alias = m.home_team
left join {{ ref('team_aliases') }} as a on a.alias = m.away_team
