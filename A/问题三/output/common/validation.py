from __future__ import annotations

from collections.abc import Iterator


EARLY_CUTOFFS = [50, 75, 100, 125, 150]
FORECAST_HORIZON = 50
SEED = 20260814


def leave_one_battery_out(battery_ids: list[int]) -> Iterator[tuple[list[int], int]]:
    ids = sorted(int(x) for x in battery_ids)
    for validation_id in ids:
        yield [battery_id for battery_id in ids if battery_id != validation_id], validation_id
