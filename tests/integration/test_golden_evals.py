"""Golden eval CI gate: attribution regressions fail the build."""

from __future__ import annotations

import pytest
from evals.golden_runner import load_scenarios, run_all


def test_golden_scenarios_exist() -> None:
    scenarios = load_scenarios()
    assert len(scenarios) >= 4  # minimum viable golden set


@pytest.mark.asyncio
async def test_all_golden_scenarios_pass() -> None:
    passed, total = await run_all()
    assert passed == total, f"{total - passed}/{total} golden scenarios failed"
    assert total >= 4
