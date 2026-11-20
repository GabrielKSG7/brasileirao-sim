-- Em cada corte e modelo, a soma das probabilidades de cada zona é o número de vagas
-- (ex.: soma de P(título) = 1; soma de P(rebaixamento) = 4).
select season, round, model,
       sum(titulo) as s_titulo, sum(g4) as s_g4, sum(rebaixamento) as s_z4
from {{ ref('stg_backtest_team_probs') }}
group by 1, 2, 3
having abs(sum(titulo) - 1) > 1e-6 or abs(sum(g4) - 4) > 1e-6 or abs(sum(rebaixamento) - 4) > 1e-6
