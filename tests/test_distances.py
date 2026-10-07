import json
import math
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from triage.distances import (
    COMMUNITIES_PATH,
    EARTH_RADIUS_KM,
    OFFICES_PATH,
    haversine_km,
    load_communities,
    load_offices,
    nearest_office,
)

RAW_COMMUNITIES: list[dict[str, Any]] = json.loads(COMMUNITIES_PATH.read_text(encoding="utf-8"))
RAW_OFFICES: list[dict[str, Any]] = json.loads(OFFICES_PATH.read_text(encoding="utf-8"))


def test_haversine_matches_published_reference() -> None:
    # Rosetta Code, "Haversine formula" task: Nashville BNA (36.12, -86.67) to LAX
    # (33.94, -118.40) is 2887.2599506071106 km with R = 6372.8 km. Rescaled to our radius.
    km = haversine_km(36.12, -86.67, 33.94, -118.40)
    assert km * 6372.8 / EARTH_RADIUS_KM == pytest.approx(2887.2599506071106, abs=1e-6)


def test_haversine_is_zero_for_the_same_point() -> None:
    assert haversine_km(-12.4, 130.8, -12.4, 130.8) == 0


# One community near each office (labels by hand from the office list).
@pytest.mark.parametrize(("community", "office"), [
    ("Darwin", "Casuarina office"),
    ("Daly River (Nauiyu)", "Palmerston office"),
    ("Pine Creek", "Katherine office"),
    ("Ali Curung", "Tennant Creek office"),
    ("Hermannsburg (Ntaria)", "Alice Springs office"),
    ("Yirrkala", "Nhulunbuy office"),
])
def test_nearest_office_near_each_office(community: str, office: str) -> None:
    found = nearest_office(community)
    assert found is not None and found[0] == office


def oracle(community: dict[str, Any]) -> tuple[str, int]:
    km = {o["town"] + " office": math.inf for o in RAW_OFFICES}
    for o in RAW_OFFICES:
        label = o["town"] + " office"
        km[label] = min(km[label], haversine_km(community["lat"], community["lon"], o["lat"], o["lon"]))
    label = min(km, key=km.__getitem__)
    return label, int(km[label] / 5 + 0.5) * 5


@given(st.sampled_from(RAW_COMMUNITIES))
def test_nearest_office_is_the_argmin_over_every_office(community: dict[str, Any]) -> None:
    assert nearest_office(community["name"]) == oracle(community)


def test_km_is_rounded_to_the_nearest_5() -> None:
    assert nearest_office("Darwin") == ("Casuarina office", 10)  # 8.4 km
    assert nearest_office("Galiwinku") == ("Nhulunbuy office", 135)


def test_same_building_offices_share_the_casuarina_label() -> None:
    labels = {o.name: o.label for o in load_offices()}
    assert labels["Greater Darwin"] == labels["Top End"] == "Casuarina office"


@pytest.mark.parametrize(("alias", "name"), [
    ("Port Keats", "Wadeye"),
    ("Oenpelli", "Gunbalanya"),
    ("Nguiu", "Wurrumiyanga"),
    ("Ntaria", "Hermannsburg (Ntaria)"),
    ("Hermannsburg", "Hermannsburg (Ntaria)"),
    ("Galiwin'ku", "Galiwinku"),
    ("Elcho Island", "Galiwinku"),
])
def test_alias_resolves_to_its_community(alias: str, name: str) -> None:
    assert nearest_office(alias) == nearest_office(name) is not None


@pytest.mark.parametrize("spelling", ["  port   KEATS ", "WADEYE", "galiwin'ku"])
def test_match_ignores_case_and_spacing(spelling: str) -> None:
    assert nearest_office(spelling) is not None


@pytest.mark.parametrize("spelling", ["Wadey", "Wadeye NT", "Port Keat", "Galiwinku Community", "Daly River", "Nauiyu", ""])
def test_unknown_or_misspelt_community_is_unknown(spelling: str) -> None:
    # No fuzzy match: a near miss is unknown, never a guess at the wrong place.
    assert nearest_office(spelling) is None


# --- loading: a bad row raises, never drops silently --------------------------------------

def write(tmp_path: Path, rows: object) -> Path:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_every_row_has_aliases_list() -> None:
    assert all(isinstance(r["aliases"], list) for r in RAW_COMMUNITIES)


@pytest.mark.parametrize(("field", "value"), [("lat", -9.5), ("lat", -26.5), ("lon", 128.9), ("lon", 138.1),
                                              ("source_url", "")])
def test_community_out_of_range_or_unsourced_fails_to_load(tmp_path: Path, field: str, value: object) -> None:
    rows = [dict(r) for r in RAW_COMMUNITIES]
    rows[5][field] = value
    with pytest.raises(ValidationError, match=field):
        load_communities(write(tmp_path, rows))


@pytest.mark.parametrize("field", ["aliases", "source_url", "lat"])
def test_community_missing_field_fails_to_load(tmp_path: Path, field: str) -> None:
    rows = [dict(r) for r in RAW_COMMUNITIES]
    del rows[0][field]
    with pytest.raises(ValidationError, match=field):
        load_communities(write(tmp_path, rows))


@pytest.mark.parametrize(("field", "value"), [("lat", 0.0), ("lon", 140.0), ("source_url", "")])
def test_office_out_of_range_or_unsourced_fails_to_load(tmp_path: Path, field: str, value: object) -> None:
    rows = [dict(r) for r in RAW_OFFICES]
    rows[2][field] = value
    with pytest.raises(ValidationError, match=field):
        load_offices(write(tmp_path, rows))


def test_alias_naming_two_communities_fails_to_load(tmp_path: Path) -> None:
    rows = [dict(r) for r in RAW_COMMUNITIES]
    rows[1]["aliases"] = ["Port Keats"]  # already Wadeye's
    with pytest.raises(ValueError, match="Port Keats"):
        load_communities(write(tmp_path, rows))


def test_offices_sharing_coordinates_must_share_a_label(tmp_path: Path) -> None:
    rows = [dict(r) for r in RAW_OFFICES]
    top_end = next(r for r in rows if r["name"] == "Top End")
    top_end["town"] = "Darwin"
    with pytest.raises(ValueError, match="share coordinates"):
        load_offices(write(tmp_path, rows))
