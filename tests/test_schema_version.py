import re
from pathlib import Path

from brasileirao_sim.ingest.footballdata_uk import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_dbt_expects_the_same_schema_version_as_ingestion():
    project = (ROOT / "dbt" / "dbt_project.yml").read_text(encoding="utf-8")
    m = re.search(r"fduk_schema_version:\s*(\d+)", project)
    assert m and int(m.group(1)) == SCHEMA_VERSION
