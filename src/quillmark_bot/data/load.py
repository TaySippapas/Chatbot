import json
import re
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_policies(path: Path) -> list[dict]:
    """Parse the handbook into one record per `### [POL-...]` block, tagged with its `##` area."""
    area, records = None, []
    for block in re.split(r"\n(?=## |### )", path.read_text(encoding="utf-8")):
        if block.startswith("### "):
            m = re.match(r"### \[(POL-[A-Z]+-\d+)\] (.+?)\n\n(.*)", block, re.S)
            records.append({"policy_id": m[1], "area": area, "title": m[2], "body": m[3].strip()})
        elif block.startswith("## "):
            area = block[3:].splitlines()[0].strip()
    return records
