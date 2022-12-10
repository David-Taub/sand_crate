import time
from collections import defaultdict

import yaml

OUTSIDE_CONTEXT = "Outside"


class Timer:
    def __init__(self) -> None:
        self.reset()
        self.ticks: int = 0

    def __call__(self, context: str = "") -> "Timer":
        self.context.append(context)
        return self

    def __enter__(self) -> "Timer":
        self.starts[self.context[-1]] = time.time()
        if len(self.context) == 1:
            self.counters[OUTSIDE_CONTEXT] += 1
            self.durations[OUTSIDE_CONTEXT] = time.time() - self.starts[OUTSIDE_CONTEXT]
        return self

    def __exit__(self, *args) -> None:
        context = self.context[-1]
        self.durations[context] += time.time() - self.starts[context]
        self.counters[self.context[-1]] += 1
        self.context.pop(-1)
        if len(self.context) == 0:
            self.starts[OUTSIDE_CONTEXT] = time.time()

    def reset(self):
        if hasattr(self, "counters"):
            self.ticks += min(self.counters.values())
        self.context = []
        self.starts = {OUTSIDE_CONTEXT: time.time()}
        self.durations = defaultdict(lambda: 0)
        self.counters = defaultdict(lambda: 0)

    def report(self) -> str:
        total_duration = sum(self.durations.values())
        contexts_report = {}
        for context, duration in self.durations.items():
            contexts_report[
                context
            ] = f"{1000 * duration / self.counters[context]:.0f} ms ({100 * duration / total_duration:.0f}%)"
        frame_duration = total_duration / min(self.counters.values())
        return yaml.dump(
            {
                "Timing": contexts_report,
                "FPS": f"{int(1 / frame_duration)} ({1000 * frame_duration:.0f} ms)",
                "Ticks": f"{self.ticks:,}",
            }
        )
