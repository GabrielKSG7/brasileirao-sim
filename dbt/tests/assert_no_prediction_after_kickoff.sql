-- Vazamento: toda previsão precisa ter sido feita com corte ANTERIOR à data do jogo.
select *
from {{ ref('stg_backtest_match_preds') }}
where cutoff >= match_date
