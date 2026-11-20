-- Falha listando os nomes de times que ainda não estão no seed team_aliases.
-- Ao rodar com dados reais pela primeira vez, é aqui que aparecem os nomes faltantes.
select distinct home_team_raw as unmapped_team from {{ ref('int_matches_unified') }} where home_team_id is null
union
select distinct away_team_raw from {{ ref('int_matches_unified') }} where away_team_id is null
