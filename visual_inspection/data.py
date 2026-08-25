"""MVTec AD dataset boundary."""

from pathlib import Path
from PIL import Image
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import v2
from torchvision import tv_tensors

@dataclass(frozen=True)
class SampleRecord:
      image_path: Path
      mask_path: Path | None
      category: str
      defect_type: str
      split: str
      label: int

def scan_records(root:Path,category:str, split:str ) -> list[SampleRecord]:
    category_path = root/category
    split_path = category_path/split

    samples = []
    for defect_path in sorted(split_path.iterdir()):
          defect_type = defect_path.name
          label = 0 if defect_path.name == "good" else 1

          for image_path in sorted(defect_path.glob("*.png")):
            mask_path = None
            if label ==1:
                mask_path = (category_path/"ground_truth"/defect_type/f"{image_path.stem}_mask.png")
                if not mask_path.exists():
                    raise FileNotFoundError(mask_path)

            samples.append(SampleRecord(
                image_path,mask_path,category,defect_type,split,label
            ))

    return samples


def load_image(path:Path):
    with Image.open(path) as img:
        return img.convert("RGB") 



def load_mask(path:Path|None, size:tuple[int,int]):
    if path is None:
        return Image.new("L", size, color = 0 )

    with Image.open(path) as img:
        mask = img.convert("L")
    if mask.size != size:
        raise ValueError("image and mask sizes do not match")

    return mask



class MVTecDataset(Dataset):
    def __init__(self, root, category, split, transform ):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform
        self.samples = scan_records(root, category, split)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        image = load_image(record.image_path)
        mask = load_mask(record.mask_path, image.size)
        image = to_tensor(image)
        mask = to_tensor(mask).squeeze(0).bool()
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        image, mask = self.transform(image, mask)




        return  {
            "image": image,
            "mask": mask,
            "label": record.label
        }


def build_transform(image_size:int):
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ]

    )
