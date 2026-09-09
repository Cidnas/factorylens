"""Small correctness checks; real GPU runs are documented separately."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from visual_inspection.model import PatchCore
from visual_inspection.storage import ExperimentStore


def evaluate_scores(scores, labels):
    # Exercise the actual evaluation method with known scores instead of a CNN.
    model = SimpleNamespace(
        device="cpu", predict=lambda images: {"images_score": images}
    )
    batches = []
    for start in range(0, len(scores), 2):
        batch_labels = labels[start:start + 2]
        batches.append({
            "image": torch.tensor(scores[start:start + 2]),
            "label": torch.tensor(batch_labels),
            "image_path": [f"sample_{i}.png" for i in range(start, start + len(batch_labels))],
            "defect_type": ["good" if label == 0 else "defect" for label in batch_labels],
        })
    return PatchCore.evaluation(model, batches)


class EvaluationTests(unittest.TestCase):
    def test_known_rankings_and_ties_across_batches(self):
        for scores, expected in [
            ([0., 1., 2., 3.], 1.),
            ([3., 2., 1., 0.], 0.),
            ([1., 1., 1., 1.], .5),
            ([0., 2., 1., 2.], .625),
        ]:
            with self.subTest(scores=scores):
                result = evaluate_scores(scores, [0, 0, 1, 1])
                self.assertEqual(result["auroc_score"], expected)

    def test_missing_class(self):
        for scores, labels in [([], []), ([1., 2.], [0, 0]), ([1., 2.], [1, 1])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_scores(scores, labels)

    def test_image_records_keep_order_and_labels(self):
        result = evaluate_scores([3., 1., 2.], [1, 0, 1])
        self.assertEqual(result["image_results"], [
            {"image_path": "sample_0.png", "defect_type": "defect", "label": 1, "image_score": 3.},
            {"image_path": "sample_1.png", "defect_type": "good", "label": 0, "image_score": 1.},
            {"image_path": "sample_2.png", "defect_type": "defect", "label": 1, "image_score": 2.},
        ])

    def test_artifact_roundtrip_and_no_overwrite(self):
        evaluation = evaluate_scores([0., 1.], [0, 1])
        result = {"config": {"seed": 42}, "metrics": {"auroc_score": evaluation["auroc_score"]}}
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory))
            path = store.save("example", result, evaluation["image_results"])
            self.assertEqual(json.loads((path / "result.json").read_text()), result)
            self.assertEqual(json.loads((path / "predictions.json").read_text()), evaluation["image_results"])
            with self.assertRaises(FileExistsError):
                store.save("example", result, [])


if __name__ == "__main__":
    unittest.main()
