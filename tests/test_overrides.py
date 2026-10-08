from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from triage.overrides import REASON_TAGS, Pin, apply_pins, pin_breaks_safety

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def pin(job_id: str, target: int, at: datetime = T0, system: int = 1) -> Pin:
    return Pin(job_id=job_id, target_position=target, reason_tag="local knowledge", by="admin", at=at,
               system_position_at_pin=system)


def order(rows) -> list[str]:
    return [r.job_id for r in rows]


# A queue as ranking hands it over: safety jobs first (sort_key's first slot).
@st.composite
def queues(draw, min_size: int = 0):
    ids = draw(st.lists(st.uuids().map(str), min_size=min_size, max_size=25, unique=True))
    levels = draw(st.lists(st.integers(0, 2), min_size=len(ids), max_size=len(ids)))
    safety = dict(zip(ids, levels))
    system = sorted(ids, key=lambda j: -min(safety[j], 1))
    arrived = {j: T0 + timedelta(hours=draw(st.integers(-48, 48))) for j in ids}
    return system, safety, arrived


@given(queues())
def test_no_pins_gives_the_system_order_exactly(q):
    system, safety, arrived = q
    rows = apply_pins(system, {}, safety, arrived)
    assert order(rows) == system
    assert [r.display_position for r in rows] == [r.system_position for r in rows] == list(range(1, len(system) + 1))
    assert not any(r.pinned or r.arrived_above_pin for r in rows)


@given(queues(min_size=1), st.data())
def test_unpinned_jobs_keep_their_relative_system_order_under_any_pins(q, data):
    system, safety, arrived = q
    chosen = data.draw(st.lists(st.sampled_from(system), unique=True))
    pins = {j: pin(j, data.draw(st.integers(1, len(system) + 5)), T0 + timedelta(minutes=i))
            for i, j in enumerate(chosen)}
    shown = order(apply_pins(system, pins, safety, arrived))
    assert [j for j in shown if j not in pins] == [j for j in system if j not in pins]
    assert sorted(shown) == sorted(system)  # nothing dropped or added


@given(queues(min_size=1), st.data())
def test_a_single_pin_lands_at_its_target_clamped_into_its_safety_group(q, data):
    system, safety, arrived = q
    job = data.draw(st.sampled_from(system))
    target = data.draw(st.integers(1, len(system) + 5))
    k = sum(safety[j] >= 1 for j in system)
    lo, hi = (1, k) if safety[job] >= 1 else (k + 1, len(system))
    row = next(r for r in apply_pins(system, {job: pin(job, target)}, safety, arrived) if r.job_id == job)
    assert row.display_position == min(max(target, lo), hi)
    assert row.pinned and row.system_position == system.index(job) + 1


@given(queues(min_size=1), st.data())
def test_pins_never_put_a_non_safety_job_above_a_safety_job(q, data):
    system, safety, arrived = q
    chosen = data.draw(st.lists(st.sampled_from(system), unique=True))
    pins = {j: pin(j, data.draw(st.integers(1, len(system))), T0 + timedelta(minutes=i)) for i, j in enumerate(chosen)}
    levels = [safety[j] >= 1 for j in order(apply_pins(system, pins, safety, arrived))]
    assert levels == sorted(levels, reverse=True)


def test_on_a_collision_the_earlier_pin_keeps_the_slot():
    system, safety = ["a", "b", "c", "d"], dict.fromkeys("abcd", 0)
    arrived = dict.fromkeys(system, T0 - timedelta(days=1))
    pins = {"d": pin("d", 1, T0), "c": pin("c", 1, T0 + timedelta(minutes=1))}
    assert order(apply_pins(system, pins, safety, arrived)) == ["d", "c", "a", "b"]
    # Same pins, earlier time swapped: the other job wins.
    pins = {"d": pin("d", 1, T0 + timedelta(minutes=1)), "c": pin("c", 1, T0)}
    assert order(apply_pins(system, pins, safety, arrived)) == ["c", "d", "a", "b"]


def test_a_collision_at_the_last_slot_takes_the_nearest_free_slot_above():
    system, safety = ["a", "b", "c"], dict.fromkeys("abc", 0)
    arrived = dict.fromkeys(system, T0)
    pins = {"a": pin("a", 3, T0), "b": pin("b", 3, T0 + timedelta(minutes=1))}
    assert order(apply_pins(system, pins, safety, arrived)) == ["c", "b", "a"]


def test_a_pin_for_a_job_no_longer_ranked_is_ignored():
    system, safety = ["a", "b"], dict.fromkeys("ab", 0)
    rows = apply_pins(system, {"gone": pin("gone", 1)}, safety, dict.fromkeys(system, T0))
    assert order(rows) == system and not any(r.pinned for r in rows)


def test_a_new_arrival_above_a_pin_is_marked_and_never_moved():
    system, safety = ["new", "old", "pinned"], dict.fromkeys(["new", "old", "pinned"], 0)
    pins = {"pinned": pin("pinned", 2, T0)}
    early = {"new": T0 - timedelta(hours=1), "old": T0 - timedelta(hours=1), "pinned": T0 - timedelta(hours=2)}
    late = {**early, "new": T0 + timedelta(hours=1)}
    before, after = apply_pins(system, pins, safety, early), apply_pins(system, pins, safety, late)
    assert order(before) == order(after) == ["new", "pinned", "old"]
    assert [r.arrived_above_pin for r in after] == [True, False, False]
    assert not any(r.arrived_above_pin for r in before)
    # Arrived after the pin but ranked below it: not marked.
    below = apply_pins(system, pins, safety, {**early, "old": T0 + timedelta(hours=1)})
    assert not any(r.arrived_above_pin for r in below)


def test_a_missing_arrival_time_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="no arrival time for b"):
        apply_pins(["a", "b"], {"a": pin("a", 2)}, dict.fromkeys("ab", 0), {"a": T0})


def test_the_safety_line_check():
    system, safety = ["s1", "s2", "n1", "n2"], {"s1": 2, "s2": 1, "n1": 0, "n2": 0}
    assert not pin_breaks_safety(system, safety, "s2", 1)  # reorder within the safety group (§3.4)
    assert pin_breaks_safety(system, safety, "s1", 3)      # safety job below a non-safety job
    assert pin_breaks_safety(system, safety, "n2", 2)      # non-safety job above a safety job
    assert not pin_breaks_safety(system, safety, "n2", 3)
    assert not pin_breaks_safety(system, safety, "n1", 99)


def test_tags_are_quick_select_only_and_the_pin_has_no_note():
    assert REASON_TAGS == ("local knowledge", "access / road", "tenant contact", "bundling with nearby job", "other")
    fields = dict(job_id="a", target_position=1, reason_tag="other", by="admin", at=T0, system_position_at_pin=2)
    assert Pin(**fields).reason_tag == "other"
    for bad in ({**fields, "reason_tag": "already made safe"}, {**fields, "note": "x"}, {**fields, "target_position": 0},
                {**fields, "at": T0.replace(tzinfo=None)}):
        with pytest.raises(ValidationError):
            Pin(**bad)
    for name in fields:  # no defaults: every field is required
        with pytest.raises(ValidationError):
            Pin(**{k: v for k, v in fields.items() if k != name})
    with pytest.raises(ValidationError):
        pin("a", 1).__setattr__("target_position", 2)  # frozen
