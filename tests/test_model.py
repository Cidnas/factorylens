from visual_inspection.model import PatchCore
from visual_inspection.data import MVTecDataset
from torch.utils.data import DataLoader
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1].resolve()
DATA_ROOT = PROJECT_ROOT/"data"/"mvtec_anomaly_detection"

## Initialzing objects 
dataset = MVTecDataset(
    root=DATA_ROOT,
    category= "bottle",
    split = "train"
    )
model = PatchCore()

dataloader = DataLoader(dataset)


c = model.test(dataloader)