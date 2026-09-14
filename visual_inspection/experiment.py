"""One controlled experiment."""
import json
import argparse
import subprocess
import time
from visual_inspection.data import MVTecDataset, build_transform
from pathlib import Path
import torch 
from dataclasses import asdict, dataclass, replace
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from visual_inspection.model import PatchCore
from visual_inspection.storage import ExperimentStore

from torchvision.models import (
      Wide_ResNet50_2_Weights,
      wide_resnet50_2,
  )

import uuid

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)

# Paths
PROJECT_ROOT = Path(__file__).parents[1].resolve()
DATA_ROOT = PROJECT_ROOT/"data"/"mvtec_anomaly_detection"
PACKAGE_ROOT = Path(__file__).parent.resolve()
CONFIG__ROOT = PACKAGE_ROOT/"configs"
TRACE_ROOT = PACKAGE_ROOT/"traces"


# define the config
@dataclass(frozen=True)
class ExperimentConfig:
    category: str
    seed: int
    backbone: str
    batch_size: int
    image_size: int
    coreset_ratio: float
    device: str


def load_config(file_name):
    file_path = CONFIG__ROOT/file_name
    with open(file_path, "r") as f:
        data = json.load(f)

    return ExperimentConfig(**data)


class ExperimentRunner:
    def __init__(self,config:ExperimentConfig, tracer ):
        self.config = config 
        self.tracer = tracer
        generator = torch.Generator().manual_seed(config.seed)
        transform= build_transform(image_size=config.image_size)
        dataset = MVTecDataset(
            root = DATA_ROOT,
            category=config.category,
            split = "train", 
            transform= transform
        )

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [0.8, 0.2],
            generator=generator
        )

        
        self.test_dataset = MVTecDataset(
            root = DATA_ROOT,
            category=config.category,
            split = "test", 
            transform=transform
        )

        self.train_dataloader= DataLoader(self.train_dataset, 
                              config.batch_size,
                              shuffle= False)

        self.val_dataloader= DataLoader(self.val_dataset, 
                            config.batch_size, 
                            shuffle=False)


        self.test_dataloader= DataLoader(self.test_dataset, 
                              config.batch_size,
                              shuffle= False)

    def _set_model(self):
        if self.config.backbone != "wide_resnet50_2":
            raise ValueError("Only wide_resnet50_2 is supported by the current embeddings.")
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        backbone.eval()

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.model = PatchCore(
            backbone,
            self.config.device,
            self.tracer,
            seed=self.config.seed,
        )



    def run(self, run_id=None):
        run_id = run_id or str(uuid.uuid4())
        with self.tracer.start_as_current_span("experiment", attributes={
            "run.id": run_id, "category": self.config.category,
            "device": self.config.device,
        }):
            return self._run(run_id)

    def _run(self, run_id):
        # The split and coreset each have their own seeded generator.
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        started = time.perf_counter()
        self._set_model()
        try:
            self.model.fit(self.train_dataloader, self.config.coreset_ratio)
            self.model.calibrate_score(self.val_dataloader)
            with self.model.telemetry.stage("evaluation") as span:
                evaluation = self.model.evaluation(self.test_dataloader)
                span.set_attribute("image.count", len(self.test_dataset))
                span.set_attribute("image.auroc", evaluation["auroc_score"])
        finally:
            cuda_stage_elapsed_ms = self.model.telemetry.finish()
        elapsed = time.perf_counter() - started
        image_results = evaluation.pop("image_results")
        result = {
            "run_id": run_id,
            "config": asdict(self.config),
            "metrics": evaluation,
            "duration_seconds": elapsed,
            "cuda_stage_elapsed_ms": cuda_stage_elapsed_ms,
            "memory_bank_size": self.model.memory_bank.shape[0],
            "threshold": self.model.threshold.item(),
            "split_sizes": {
                "train": len(self.train_dataset),
                "validation": len(self.val_dataset),
                "test": len(self.test_dataset),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()),
            "torch_version": str(torch.__version__),
            "device_name": (torch.cuda.get_device_name(self.config.device)
                            if torch.device(self.config.device).type == "cuda" else "cpu"),
        }
        store = ExperimentStore(PROJECT_ROOT / "runs")
        output_path = store.save(run_id, result, image_results)
        print(f"Image AUROC: {evaluation['auroc_score']:.6f}")
        print(f"Saved run: {output_path}")
        return result
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run one MVTec experiment.")
    parser.add_argument("--config", default="patchcore__baseline.json")
    parser.add_argument("--category")
    parser.add_argument("--coreset-ratio", type=float)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.category is not None:
        config = replace(config, category=args.category)
    if args.coreset_ratio is not None:
        config = replace(config, coreset_ratio=args.coreset_ratio)
    provider = TracerProvider()
    run_id = str(uuid.uuid4())
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    trace_file = open(TRACE_ROOT/f"traces_{run_id}", "w")
    exporter = ConsoleSpanExporter(out=trace_file)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)
    experimentrunner = ExperimentRunner(config =config, tracer= tracer )
    try:
        experimentrunner.run(run_id=run_id)
    finally:
        provider.shutdown()
        trace_file.close()
