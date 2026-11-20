-- Falha se a odd média de fechamento do mercado estiver ausente ou for <= 1
-- em jogo já disputado (a média cobre todas as temporadas).
select *
from {{ ref('stg_fduk_matches') }}
where is_played
  and (odds_avgch is null or odds_avgcd is null or odds_avgca is null
       or odds_avgch <= 1 or odds_avgcd <= 1 or odds_avgca <= 1)
