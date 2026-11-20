-- Em pontos corridos com 20 times, cada time joga no máximo 19 vezes em casa e 19 fora por temporada.
with games as (
    select season, home_team as team, 'home' as side from {{ ref('stg_fduk_matches') }}
    union all
    select season, away_team, 'away' from {{ ref('stg_fduk_matches') }}
)
select season, team, side, count(*) as n
from games
group by 1, 2, 3
having count(*) > 19
