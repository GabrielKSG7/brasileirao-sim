-- Formato longo: uma linha por (temporada, corte, modelo, time, zona) com a probabilidade
-- prevista e o que aconteceu. Só temporadas completas.
with p as (
    select * from {{ ref('stg_backtest_team_probs') }}
),
f as (
    select * from {{ ref('stg_final_positions') }}
),
long as (
    {% set zones = ['titulo', 'g4', 'libertadores', 'sul_americana', 'rebaixamento'] %}
    {% for z in zones %}
    select
        p.season, p.round, p.cutoff, p.model, p.team_id,
        '{{ z }}' as zona,
        p.{{ z }} as prob,
        case when f.final_{{ z }} then 1 else 0 end as aconteceu
    from p
    join f on f.season = p.season and f.team_id = p.team_id
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
)
select * from long
