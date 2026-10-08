"""Stage 5 static tables: road distance from Darwin, and community coordinates for a labelled
straight-line estimate when no road distance is known. Logistics display only, never read by ranking."""

import json
import math
from pathlib import Path
from typing import Optional

DATA = Path(__file__).resolve().parent.parent / "data"
DISTANCES_PATH = DATA / "distances.json"
COMMUNITIES_PATH = DATA / "communities.json"


def load_distances(path: Path = DISTANCES_PATH) -> dict[str, float]:
    """Community name -> distance in km."""
    return {name: float(km) for name, km in json.loads(path.read_text(encoding="utf-8")).items()}


def load_coordinates(path: Path = COMMUNITIES_PATH) -> dict[str, tuple[float, float]]:
    """Community name -> (lat, lon). Keys starting with "_" are notes, not communities."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: (float(v[0]), float(v[1])) for name, v in raw.items() if not name.startswith("_")}


def straight_line_km(a: str, b: str, coords: dict[str, tuple[float, float]]) -> Optional[float]:
    """Great-circle km between two communities, or None if either is not in the table."""
    if a not in coords or b not in coords:
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, (*coords[a], *coords[b]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return float(round(6371 * 2 * math.asin(math.sqrt(h))))
