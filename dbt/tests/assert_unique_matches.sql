-- Falha se existir mais de um jogo para (data, mandante, visitante).
select match_date, home_team, away_team, count(*) as n
from {{ ref('stg_fduk_matches') }}
group by 1, 2, 3
having count(*) > 1
