-- Jogos com nomes de times padronizados (seed team_aliases).
-- Nome sem correspondência fica com team_id nulo e é pego pelo teste assert_no_unmapped_teams.
select
    m.season,
    m.match_date,
    m.kickoff_utc,
    m.home_team as home_team_raw,
    m.away_team as away_team_raw,
    h.team_id as home_team_id,
    a.team_id as away_team_id,
    m.home_goals,
    m.away_goals,
    m.result,
    m.is_played,
    m.odds_psch, m.odds_pscd, m.odds_psca,
    m.odds_avgch, m.odds_avgcd, m.odds_avgca,
    m.odds_maxch, m.odds_maxcd, m.odds_maxca
from {{ ref('stg_fduk_matches') }} as m
left join {{ ref('team_aliases') }} as h on h.alias = m.home_team
left join {{ ref('team_aliases') }} as a on a.alias = m.away_team
