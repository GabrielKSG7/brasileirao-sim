-- Calibração jogo a jogo: probabilidade de vitória do mandante em faixas de 10 p.p.
select
    model,
    least(floor(p_home * 10), 9) as faixa,
    count(*) as n,
    avg(p_home) as prob_media,
    avg(case when result = 'H' then 1.0 else 0.0 end) as freq_observada
from {{ ref('stg_backtest_match_preds') }}
where p_home is not null
group by 1, 2
