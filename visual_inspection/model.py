from torchvision.models import (
      feature_extraction,
      Wide_ResNet50_2_Weights,
      wide_resnet50_2,
  )
import torch
import torch.nn as nn
import torch.nn.functional as F


backbone = wide_resnet50_2(
      weights=Wide_ResNet50_2_Weights.DEFAULT
  )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.eval()

for parameter in backbone.parameters():
      parameter.requires_grad = False




class PatchCore(nn.Module):
      def __init__(self, backbone=backbone, device=device, threshold = None):
            super().__init__()
            self.backbone = backbone.to(device)
            self.device = device
            self.threshold = threshold
            self.register_buffer("memory_bank", None)
            self.feature_extractor= feature_extraction.create_feature_extractor(
                  model=self.backbone,
                  return_nodes={
                  "layer2": "layer2",
                  "layer3": "layer3"
            }
            ).to(device)


      def make_embeddings(self, features):
            unfolder = nn.Unfold(
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                  )
            shape2, shape3 = features["layer2"].shape, features["layer3"].shape
            features_2, features_3= unfolder(features["layer2"]), unfolder(features["layer3"])

            features_3 = features_3.unflatten(-1, shape3[-2:])

            # interpolating layer3 features to match layer2 features
            features_3 = F.interpolate(features_3, 
                                          size = shape2[-2:],
                                          mode='bilinear', 
                                          align_corners=False)

            features_3 = features_3.flatten(start_dim =2)

            assert features_2.shape[-1] == features_3.shape[-1], \
            f"Patch count mismatch: {features_2.shape[-1]} != {features_3.shape[-1]}"
            features_2, features_3= features_2.transpose(1,2), features_3.transpose(1,2)
            B,P, emb2 = features_2.shape
            emb3 = features_3.shape[2]
            features_2, features_3 = features_2.reshape(B*P, 1, emb2), features_3.reshape(B*P, 1, emb3)

            features_2, features_3= F.adaptive_avg_pool1d(features_2, output_size=1024), \
                                    F.adaptive_avg_pool1d(features_3, output_size=1024)

            features_2, features_3 = features_2.squeeze(1), features_3.squeeze(1)

            embds = torch.stack([features_2, features_3], dim=1)
            embds = embds.reshape(B*P, 1, 2048)
            embds = F.adaptive_avg_pool1d(embds, output_size=1024)
            embds = embds.squeeze(1)
            embds = embds.reshape(B,P,1024)

            return embds

      def coreset(self, embds):

            ## To be implemented
            return embds


      def embd_score(self, embds):
            """
            Comparing the embds with the embds in the memory bank
            """
            distances = torch.cdist(
                  x1= embds, 
                  x2 = self.memory_bank,
                  p=2
            )

            patch_scores  = torch.amin(distances, dim= -1)

            return patch_scores

      def batch_embds(self, dataloader):            
            embd_batches = []
            with torch.inference_mode():
                  for batch in dataloader:
                        images = batch["image"].to(self.device)
                        features = self.feature_extractor(images)
                        embds = self.make_embeddings(features)
                        embd_batches.append(embds.detach().cpu())
            
            embd_batches = torch.cat(embd_batches, dim=0)
            return embd_batches


      def calibrate_score(self,val_dataloader):
            embd_batches = self.batch_embds(val_dataloader)
            patches_score = self.embd_score(embd_batches)
            images_score = torch.amax(patches_score, dim=-1)
            p99 = torch.quantile(images_score, 0.99 )
            self.threshold = p99





      def fit(self, train_dataloader):
            embd_batches = self.batch_embds(train_dataloader)
            N,P,D = embd_batches.shape
            embd_batches =  embd_batches.reshape(N*P, D)
            self.memory_bank = self.coreset(embd_batches).to(device)


      def predict(self, batch):
            embds = self.make_embeddings(batch)

            patch_score = self.embd_score(embds)
            image_score = patch_score.amax(patch_score, dim=-1)


            ## Placeholder

            if image_score > self.threshold: 
                  anomaly_map = self.anomaly(batch)

            return {
                  "patch_score": patch_score,
                  "image_score": image_score,
                  "anomaly_map": anomaly_map
            }

            


            
                  







