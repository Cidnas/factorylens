from visual_inspection.model import PatchCore
from visual_inspection.data import MVTecDataset, build_transform
from torch.utils.data import DataLoader
from pathlib import Path
import torch 

PROJECT_ROOT = Path(__file__).parents[1].resolve()
DATA_ROOT = PROJECT_ROOT/"data"/"mvtec_anomaly_detection"


# test config 
ratio = 0.01

## Initialzing objects 
transform= build_transform(image_size=224)
device = torch.device("cuda:0")
dataset = MVTecDataset(
    root=DATA_ROOT,
    category= "bottle",
    split = "train",
    transform=transform

    )
model = PatchCore()

dataloader = DataLoader(dataset)

model.fit(dataloader, ratio)


print(f"Final Debug message:{type(model.memory_bank)}")