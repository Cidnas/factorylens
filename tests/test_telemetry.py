"""Check deferred span export, stage boundaries, and errors on CPU."""

import time
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
import torch

from visual_inspection.model import PatchCore
from visual_inspection.telemetry import StageTracer


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.provider = TracerProvider()
        self.exporter = InMemorySpanExporter()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = self.provider.get_tracer(__name__)
        self.stages = StageTracer(self.tracer, "cpu")
        self.addCleanup(self.provider.shutdown)

    def test_cpu_stage_keeps_original_boundary_and_parent(self):
        with patch("torch.cuda.Event", side_effect=AssertionError("CPU created CUDA event")):
            with self.tracer.start_as_current_span("experiment") as parent:
                with self.stages.stage("example") as child:
                    child.set_attribute("image.count", 2)
                stage_finished = time.time_ns()
                self.assertEqual(len(self.exporter.get_finished_spans()), 0)
                self.stages.finish()
                self.stages.finish()  # No duplicate export.
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        stage = spans[0]
        self.assertEqual(stage.parent.span_id, parent.get_span_context().span_id)
        self.assertLessEqual(stage.end_time, stage_finished)
        self.assertEqual(stage.attributes["image.count"], 2)
        self.assertNotIn("cuda.elapsed_ms", stage.attributes)
        self.assertEqual(self.stages.pending, [])

    def test_stage_failure_is_exported_and_propagates(self):
        with self.assertRaisesRegex(ValueError, "bad input"):
            with self.tracer.start_as_current_span("experiment"):
                try:
                    with self.stages.stage("evaluation"):
                        raise ValueError("bad input")
                finally:
                    self.stages.finish()
        stage, parent = self.exporter.get_finished_spans()
        for span in (stage, parent):
            self.assertEqual(span.status.status_code, StatusCode.ERROR)
            self.assertTrue(any(event.name == "exception" for event in span.events))

    def test_trace_ids_do_not_change_coreset_selection(self):
        embeddings = torch.tensor([[0., 0.], [1., 0.], [4., 0.], [10., 0.]])
        first = PatchCore.coreset(
            SimpleNamespace(coreset_rng=random.Random(42)), embeddings, .75)
        for _ in range(5):
            self.tracer.start_span("extra instrumentation").end()
        second = PatchCore.coreset(
            SimpleNamespace(coreset_rng=random.Random(42)), embeddings, .75)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
