-- Probabilidades jogo a jogo entre 0 e 1 e somando 1.
select *
from {{ ref('stg_backtest_match_preds') }}
where p_home is not null
  and (abs(p_home + p_draw + p_away - 1) > 1e-6
       or least(p_home, p_draw, p_away) < 0)
