"""Straight-line distance from a community to its nearest NT housing office, for display.

Input: data/communities.json and data/housing_offices.json (sourced coordinates). Output:
(office label, km rounded to 5) for a community, or None when the community isn't in the
table. Never read by ranking (invariant 5): distance must not move a job up or down the queue.
"""

import json
import math
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMMUNITIES_PATH = DATA_DIR / "communities.json"
OFFICES_PATH = DATA_DIR / "housing_offices.json"

EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius
# The NT's extent with margin: a coordinate outside it is a data error, not a remote place.
NtLat = Annotated[float, Field(ge=-26, le=-10)]
NtLon = Annotated[float, Field(ge=129, le=138)]


class Community(BaseModel):
    """One data/communities.json row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    lat: NtLat
    lon: NtLon
    source_url: str = Field(min_length=1)
    # No default: every row must say which other names its source page confirms ([] for none).
    aliases: tuple[str, ...]


class Office(BaseModel):
    """One data/housing_offices.json row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    region: str = Field(min_length=1)
    town: str = Field(min_length=1)
    address: str = Field(min_length=1)
    lat: NtLat
    lon: NtLon
    coord_level: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    office_list_source: str = Field(min_length=1)

    @property
    def label(self) -> str:
        # By town, so Greater Darwin and Top End (same building, Casuarina) share one label.
        return f"{self.town} office"


def _key(name: str) -> str:
    # Exact match only, ignoring case and spacing; no fuzzy matching, so a misspelling is unknown.
    return " ".join(name.split()).casefold()


def load_communities(path: Path = COMMUNITIES_PATH) -> dict[str, Community]:
    """Normalised name or alias -> community.

    Raises:
        ValueError: a row fails validation, or a name or alias belongs to two rows.
    """
    by_key: dict[str, Community] = {}
    for raw in json.loads(path.read_text(encoding="utf-8")):
        row = Community.model_validate(raw)
        for name in (row.name, *row.aliases):
            other = by_key.setdefault(_key(name), row)
            if other is not row:
                raise ValueError(f"{path.name}: {name!r} names both {other.name!r} and {row.name!r}")
    return by_key


def load_offices(path: Path = OFFICES_PATH) -> tuple[Office, ...]:
    """Every office row.

    Raises:
        ValueError: a row fails validation, or two offices share coordinates but not a label.
    """
    offices = tuple(Office.model_validate(r) for r in json.loads(path.read_text(encoding="utf-8")))
    by_point: dict[tuple[float, float], str] = {}
    for o in offices:
        label = by_point.setdefault((o.lat, o.lon), o.label)
        if label != o.label:
            raise ValueError(f"{path.name}: {label!r} and {o.label!r} share coordinates")
    return offices


COMMUNITIES = load_communities()
OFFICES = load_offices()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points given in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def nearest_office(community: str) -> tuple[str, int] | None:
    """(office label, straight-line km rounded to the nearest 5) for a community, or None if unknown."""
    row = COMMUNITIES.get(_key(community))
    if row is None:
        return None
    office, km = min(((o, haversine_km(row.lat, row.lon, o.lat, o.lon)) for o in OFFICES), key=lambda p: p[1])
    # Half up, not Python's round-half-even; 5 km steps so the figure doesn't look survey-exact.
    return office.label, int(km / 5 + 0.5) * 5


def km_between(a: str, b: str) -> float | None:
    """Straight-line km between two listed communities (e.g. a tradie's base and a job), or None
    if either is not in the table. Display only, like nearest_office."""
    ra, rb = COMMUNITIES.get(_key(a)), COMMUNITIES.get(_key(b))
    if ra is None or rb is None:
        return None
    return float(int(haversine_km(ra.lat, ra.lon, rb.lat, rb.lon) / 5 + 0.5) * 5)


def community_names() -> list[str]:
    """Every listed community name (not aliases), for intake suggestions."""
    return sorted({row.name for row in COMMUNITIES.values()})
