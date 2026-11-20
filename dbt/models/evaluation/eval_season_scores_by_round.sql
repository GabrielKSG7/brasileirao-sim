-- Como a precisão melhora ao longo da temporada: Brier por rodada, modelo e zona,
-- na média das temporadas completas.
select
    round,
    model,
    zona,
    count(distinct season) as temporadas,
    avg(power(prob - aconteceu, 2)) as brier
from {{ ref('eval_season_long') }}
group by 1, 2, 3
