-- Um registro por clube canônico.
select distinct team_id, team_name
from {{ ref('team_aliases') }}
