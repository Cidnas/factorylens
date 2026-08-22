"""One controlled experiment."""
import json
from visual_inspection.data import MVTecDataset
from pathlib import Path

from dataclasses import dataclass

# Paths
PROJECT_ROOT = Path(__file__).parents[1].resolve()
DATA_ROOT = PROJECT_ROOT/"data"/"mvtec_anomaly_detection"
PACKAGE_ROOT = Path(__file__).parent.resolve()
CONFIG__ROOT = PACKAGE_ROOT/"configs"


# define the config
@dataclass(frozen=True)
class ExperimentConfig:
    category: str
    seed: int
    backbone: str
    batch_size: int



def load_config(file_name):
    file_path = CONFIG__ROOT/file_name
    with open(file_path, "r") as f:
        data = json.load(f)

    return ExperimentConfig(**data)


class ExperimentRunner:
    def __init__(self,config:ExperimentConfig ):
        self.config = config 
        self.train_dataset = MVTecDataset(
            root = DATA_ROOT,
            category=config.category,
            split = "train"
        )
        self.test_dataset = MVTecDataset(
            root = DATA_ROOT,
            category=config.category,
            split = "test"
        )

        self.train_dataloader(self.train_dataset, 
                              config.batch_size,
                              shuffle= False)


        self.test_dataloader(self.test_dataset, 
                              config.batch_size,
                              shuffle= False)
    def run(self):


        pass
        

