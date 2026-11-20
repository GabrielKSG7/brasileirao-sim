-- Falha se o resultado (H/D/A) não bater com o placar.
select *
from {{ ref('stg_fduk_matches') }}
where is_played
  and result <> case
        when home_goals > away_goals then 'H'
        when home_goals < away_goals then 'A'
        else 'D'
      end
