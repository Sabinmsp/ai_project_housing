"""Coordinator overrides: pins laid over the ranked queue for display (master §5.1, FR2y).

Pipeline: intake -> extraction -> verification -> evaluation -> ranking -> overrides -> explain.
Input: the system order from ranking plus the coordinator's pins. Output: one DisplayRow per
ranked job. Pins never enter ranking: the system order is taken as given and never re-sorted.
Pure functions only, no I/O.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Optional, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

# Master §5.1: quick-select tags, never free text. No "already made safe": a made-safe job needs
# its safety status changed, not a reorder.
ReasonTag = Literal["local knowledge", "access / road", "tenant contact", "bundling with nearby job", "other"]
REASON_TAGS: tuple[str, ...] = get_args(ReasonTag)
SAFETY_BLOCK = "Safety jobs stay above the queue. Dispatch any job directly from Assign instead."


class Pin(BaseModel):
    """One coordinator pin: where the job should show, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    target_position: int = Field(ge=1)
    reason_tag: ReasonTag
    by: str
    at: AwareDatetime
    system_position_at_pin: int = Field(ge=1)


class DisplayRow(BaseModel):
    """One ranked job as the coordinator sees it: both positions, and whether it is pinned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    display_position: int = Field(ge=1)
    system_position: int = Field(ge=1)
    pinned: bool
    pin: Optional[Pin]
    arrived_above_pin: bool


def _is_safety(level: int) -> bool:
    # Master §3.4: conditional and active form one group the coordinator may reorder by pin.
    return level >= 1


def pin_breaks_safety(system_order: list[str], safety: Mapping[str, int], job_id: str, target: int) -> bool:
    """True if pinning job_id at target would put a non-safety job above a safety job, or the reverse."""
    # ranking.sort_key puts safety level first, so safety jobs hold positions 1..k.
    k = sum(_is_safety(safety[j]) for j in system_order)
    return target > k if _is_safety(safety[job_id]) else target <= k


def apply_pins(system_order: list[str], pins: Mapping[str, Pin], safety: Mapping[str, int],
               arrived: Mapping[str, datetime]) -> list[DisplayRow]:
    """The display order: pinned jobs at their target, everyone else in system order.

    A pin is clamped into the slots its safety group holds, so a pin can never move a job
    across the safety line, even after new jobs arrive. Two pins on one slot: the earlier
    pin keeps it and the later takes the nearest free slot below, else above. Pins for jobs
    not in system_order (closed or dispatched) are ignored.

    Raises:
        ValueError: a job has no arrival time while a pin is active.
    """
    system_pos = {job_id: i for i, job_id in enumerate(system_order, start=1)}
    active = sorted((p for p in pins.values() if p.job_id in system_pos), key=lambda p: (p.at, p.job_id))
    if active:
        missing = [j for j in system_order if j not in arrived]
        if missing:
            raise ValueError(f"no arrival time for {', '.join(missing)}: cannot mark new arrivals above a pin")
    slot_of: dict[str, int] = {}
    for group in (True, False):
        slots = [system_pos[j] for j in system_order if _is_safety(safety[j]) is group]
        free = list(slots)
        for pin in (p for p in active if _is_safety(safety[p.job_id]) is group):
            # The nearest free slot at or below the target, else the nearest above: this also
            # clamps a target outside the group's slots to the group's first or last slot.
            below = [s for s in free if s >= pin.target_position]
            chosen = below[0] if below else free[-1]
            free.remove(chosen)
            slot_of[pin.job_id] = chosen
        unpinned = [j for j in system_order if _is_safety(safety[j]) is group and j not in slot_of]
        slot_of.update(zip(unpinned, free, strict=True))

    by_job = {p.job_id: p for p in active}
    rows = []
    for job_id in sorted(system_order, key=slot_of.__getitem__):
        pin = by_job.get(job_id)
        # Master §5.1: a new arrival ranking above a pin is marked, never moved.
        above = pin is None and any(system_pos[job_id] < slot_of[p.job_id] and arrived[job_id] > p.at
                                    for p in active)
        rows.append(DisplayRow(job_id=job_id, display_position=slot_of[job_id], system_position=system_pos[job_id],
                               pinned=pin is not None, pin=pin, arrived_above_pin=above))
    return rows
