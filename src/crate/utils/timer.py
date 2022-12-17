import time
from collections import defaultdict

import yaml

OUTSIDE_CONTEXT = "Outside"
DECAY = 0.9


class Timer:
    def __init__(self) -> None:
        self.context = []
        self.starts = {OUTSIDE_CONTEXT: time.time()}
        self.durations = defaultdict(lambda: 0)

    def __call__(self, context: str = "") -> "Timer":
        self.context.append(context)
        return self

    def __enter__(self) -> "Timer":
        self.starts[self.context[-1]] = time.time()
        if len(self.context) == 1:
            self._update_duration(OUTSIDE_CONTEXT, time.time() - self.starts[OUTSIDE_CONTEXT])
        return self

    def __exit__(self, *args) -> None:
        context = self.context[-1]
        self._update_duration(context, time.time() - self.starts[context])
        self.context.pop(-1)
        if len(self.context) == 0:
            self.starts[OUTSIDE_CONTEXT] = time.time()

    def _update_duration(self, context: str, duration: float):
        self.durations[context] *= DECAY
        self.durations[context] += (1 - DECAY) * duration

    def report(self) -> str:
        frame_duration = sum(self.durations.values())
        contexts_report = {}
        for context, duration in self.durations.items():
            contexts_report[context] = \
                f"{1000 * duration :.0f} ms ({100 * duration / frame_duration:.0f}%)"
        return yaml.dump(
            {
                "Timing": contexts_report,
                "FPS": f"{int(1 / frame_duration)} ({1000 * frame_duration:.0f} ms)",
            }
        )
