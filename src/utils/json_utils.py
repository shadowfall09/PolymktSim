import json
from datetime import datetime
from pathlib import Path


def _default(o: object):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(type(o).__name__)


def load_json(path: str | Path) -> object:
    with open(path) as f:
        return json.load(f)


def save_json(path: str | Path, obj: object) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, default=_default, indent=2)
