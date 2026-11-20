-- Falha se um par (mandante, visitante) aparecer mais de uma vez na mesma temporada.
select season, home_team, away_team, count(*) as n
from {{ ref('stg_fduk_matches') }}
group by 1, 2, 3
having count(*) > 1
