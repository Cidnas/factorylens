### Tests for the data module


from pathlib import Path

from anomaly_detection.data import MVTecDataset
import matplotlib.pyplot as plt 

PROJECT_ROOT = Path(__file__).parents[1].resolve()
DATA_ROOT = PROJECT_ROOT/"data"/"mvtec_anomaly_detection"

dataset = MVTecDataset(
    root=DATA_ROOT,
    category= "bottle",
    split = "test"
    )

sample = dataset[3]

print(len(dataset))
print(sample["image"].shape)
print(sample["image"].dtype)
print(sample["mask"].shape)
print(sample["mask"].dtype)
print(sample["label"])


image = sample["image"].permute(1,2,0)
mask = sample["mask"]


figures,axes = plt.subplots(1,3,figsize=(12, 4))

axes[0].imshow(image)
axes[0].set_title("Image")


axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Ground-truth mask")

axes[2].imshow(image)
axes[2].imshow(mask, cmap="Reds", alpha=0.2)
axes[2].set_title("Defect over image")

for axis in axes:
      axis.axis("off")
plt.show()