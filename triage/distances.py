"""The static NT distance table (km from Darwin): logistics display only, never read by ranking."""

import json
from pathlib import Path

DISTANCES_PATH = Path(__file__).resolve().parent.parent / "data" / "distances.json"


def load_distances(path: Path = DISTANCES_PATH) -> dict[str, float]:
    """Community name -> distance in km."""
    return {name: float(km) for name, km in json.loads(path.read_text(encoding="utf-8")).items()}
