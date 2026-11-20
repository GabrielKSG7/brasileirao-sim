-- Brier e log loss das probabilidades por zona, por temporada e modelo (menor é melhor).
select
    season,
    model,
    zona,
    count(*) as previsoes,
    avg(power(prob - aconteceu, 2)) as brier,
    avg(-(aconteceu * ln(least(greatest(prob, 1e-4), 1 - 1e-4))
          + (1 - aconteceu) * ln(1 - least(greatest(prob, 1e-4), 1 - 1e-4)))) as log_loss
from {{ ref('eval_season_long') }}
group by 1, 2, 3
