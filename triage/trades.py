"""Stage 5 reference data: the trade a fault usually needs (display only, never ranks).

Keyed by TIER_TABLE names, so every fault the model can match has a trade; a test pins that.
"""

TRADE_FOR_FAULT: dict[str, tuple[str, ...]] = {
    "blocked or broken toilet": ("Plumber",),
    "blocked drain": ("Plumber",),
    "sewage leak": ("Plumber",),
    "leaking or burst water main or pipe": ("Plumber",),
    "exposed electrical wires": ("Electrician",),
    "gas leak": ("Gas fitter",),
    "roof leak": ("Roofer",),
    "flooding or flood damage": ("Plumber",),
    "storm, fire or impact damage": ("Builder",),
    "no gas, electricity or water supply": ("Electrician", "Plumber"),
    "hot water system not working": ("Plumber",),
    "stove or oven not working": ("Electrician",),
    "dripping tap or tap tight to turn": ("Plumber",),
    "stove element not working": ("Electrician",),
    "fan not working properly": ("Electrician",),
    "power point not working": ("Electrician",),
}

ALL_TRADES = ("Plumber", "Electrician", "Gas fitter", "Roofer", "Builder", "General maintenance")


def required_trades(tier_entry: str | None) -> list[str]:
    """Trades for the fault that produced the tier; empty when the job is untiered."""
    return list(TRADE_FOR_FAULT.get(tier_entry, ())) if tier_entry else []
