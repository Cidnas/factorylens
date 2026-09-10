"""Stage spans with CUDA timings collected once at the end of a run."""

from contextlib import contextmanager
import time

from opentelemetry import trace
import torch


class StageTracer:
    def __init__(self, tracer, device):
        self.tracer = tracer
        self.device = torch.device(device)
        self.pending = []

    @contextmanager
    def stage(self, name):
        span = self.tracer.start_span(name)
        start = end = None
        try:
            with trace.use_span(span, end_on_exit=False):
                if self.device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(torch.cuda.current_stream(self.device))
                yield span
        finally:
            if end is not None:
                end.record(torch.cuda.current_stream(self.device))
            # Preserve the stage boundary, even though export happens later.
            self.pending.append((name, span, start, end, time.time_ns()))

    def finish(self):
        timings = {}
        try:
            if self.pending and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            for name, span, start, end, _ in self.pending:
                if start is not None:
                    timings[name] = start.elapsed_time(end)
                    span.set_attribute("cuda.elapsed_ms", timings[name])
        finally:
            for _, span, _, _, end_time in self.pending:
                span.end(end_time=end_time)
            self.pending.clear()
        return timings
