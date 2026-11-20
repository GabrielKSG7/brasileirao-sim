-- Calibração: previsões agrupadas em faixas de 10 pontos percentuais.
-- Se o modelo é calibrado, prob_media ~ freq_observada em cada faixa.
-- Exclui probabilidades 0 e 1 exatas (eventos já decididos), que inflariam as pontas.
select
    model,
    zona,
    least(floor(prob * 10), 9) as faixa,
    count(*) as n,
    avg(prob) as prob_media,
    avg(aconteceu) as freq_observada
from {{ ref('eval_season_long') }}
where prob > 0 and prob < 1
group by 1, 2, 3
