-- Métricas jogo a jogo por temporada e modelo (menor é melhor).
-- RPS com 3 resultados ordenados (mandante, empate, visitante); log loss com piso de 1e-4.
with scored as (
    select
        season,
        model,
        case result when 'H' then 1 else 0 end as o_h,
        case result when 'D' then 1 else 0 end as o_d,
        case result when 'A' then 1 else 0 end as o_a,
        p_home, p_draw, p_away
    from {{ ref('stg_backtest_match_preds') }}
    where p_home is not null
)
select
    season,
    model,
    count(*) as jogos,
    avg(0.5 * (power(p_home - o_h, 2) + power(p_home + p_draw - o_h - o_d, 2))) as rps,
    avg(-ln(greatest(p_home * o_h + p_draw * o_d + p_away * o_a, 1e-4))) as log_loss,
    avg(power(p_home - o_h, 2) + power(p_draw - o_d, 2) + power(p_away - o_a, 2)) as brier
from scored
group by 1, 2
