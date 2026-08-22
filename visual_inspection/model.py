from torchvision.models import (
      Wide_ResNet50_2_Weights,
      wide_resnet50_2,
  )



backbone = wide_resnet50_2(
      weights=Wide_ResNet50_2_Weights.DEFAULT
  )


backbone.eval()

for parameter in backbone.parameters():
      parameter.requires_grad = False



