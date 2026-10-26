"""Shared test setup.

Unit tests need nothing but this repo. Integration tests (navmap, scene export) need the official
microduck_rl checkout for its scenes and meshes — found the same way run.sh finds it ($DUCK_SIM_RL,
else ~/Pollen/microduck_rl, else next to this repo) — and are skipped when it is missing.
Only `mujoco`, `numpy`, `scipy` and `Pillow` are imported from it, never the training stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _find_microduck_rl() -> Path | None:
    candidates = [os.environ.get("DUCK_SIM_RL"), Path.home() / "Pollen" / "microduck_rl", ROOT.parent / "microduck_rl"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "src" / "mjlab_microduck").is_dir():
            return Path(candidate)
    return None


MICRODUCK_RL = _find_microduck_rl()
if MICRODUCK_RL is not None:
    sys.path.insert(0, str(MICRODUCK_RL / "src"))

needs_scenes = pytest.mark.skipif(MICRODUCK_RL is None, reason="microduck_rl checkout not found (set DUCK_SIM_RL)")


class FakeClock:
    """Stands in for time.monotonic so gesture tests run in milliseconds, not seconds."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("time.monotonic", fake)
    return fake


class FakeRunner:
    """Records what would have been executed instead of moving a duck."""

    def __init__(self):
        self.submitted: list = []
        self.cancelled = 0
        self.current = None
        self.current_source = None
        self.log: list[str] = []
        self.actions = FakeActions()

    def submit(self, steps) -> None:
        self.submitted += list(steps)

    def cancel(self) -> None:
        self.cancelled += 1

    def busy(self) -> bool:
        return False

    def names(self) -> list[str]:
        return [step.name for step in self.submitted]


class FakeActions:
    def __init__(self):
        self.queried: list = []

    def run(self, step):
        self.queried.append(step)
        return "现在站着", 0.0

    def get_status(self):
        return "现在站着", 0.0


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()
