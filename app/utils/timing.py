"""High-resolution timing utilities for measuring pipeline latency."""

import time
from contextlib import contextmanager
from typing import Dict, Generator


class StageTimer:
    """Tracks latency across multiple pipeline stages using high-resolution timers."""

    def __init__(self) -> None:
        self._start_time: float = time.perf_counter()
        self._stages: Dict[str, float] = {}

    @contextmanager
    def measure(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager to measure execution time of a specific stage in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._stages[stage_name] = round(elapsed_ms, 2)

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Manually record a stage duration in milliseconds."""
        self._stages[stage_name] = round(duration_ms, 2)

    @property
    def total_elapsed_ms(self) -> float:
        """Total elapsed time since timer creation in milliseconds."""
        return round((time.perf_counter() - self._start_time) * 1000.0, 2)

    @property
    def stages(self) -> Dict[str, float]:
        """Dictionary of recorded stage durations."""
        timings = dict(self._stages)
        timings["total_ms"] = self.total_elapsed_ms
        return timings
