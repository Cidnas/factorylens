"""One controlled experiment."""
import json
from visual_inspection.data import MVTecDataset, build_transform
from pathlib import Path
import torch 
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from visual_inspection.model import PatchCore

from torchvision.models import (
      feature_extraction,
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
        # TODO: create backbone from config
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        backbone.eval()

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.model = PatchCore(
            backbone,
            self.config.device,
            self.tracer
        )



    def run(self):
        self._set_model()
        self.model.fit(self.train_dataloader, self.config.coreset_ratio)


        
        

if __name__=="__main__":
    config = load_config(file_name=CONFIG__ROOT/"patchcore__baseline.json")
    provider = TracerProvider()
    run_id =  uuid.uuid4()
    trace_file = open(TRACE_ROOT/f"traces_{run_id}", "w")
    exporter = ConsoleSpanExporter(out=trace_file)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)
    experimentrunner = ExperimentRunner(config =config, tracer= tracer )
    experimentrunner.run()

    trace_file.close()
