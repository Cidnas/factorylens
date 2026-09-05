from torchvision.models import (
      feature_extraction,
      Wide_ResNet50_2_Weights,
      wide_resnet50_2,
  )
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import time
from tqdm import tqdm 

from opentelemetry import trace 
from opentelemetry.sdk.trace.export import ConsoleSpanExporter


# backbone: nn.Module; input images are floating tensors shaped (B, 3, H, W).
backbone = wide_resnet50_2(
      weights=Wide_ResNet50_2_Weights.DEFAULT
  )

# device: torch.device; parameters, inputs, and produced features must share it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.eval()

for parameter in backbone.parameters():
      parameter.requires_grad = False




class PatchCore(nn.Module):
      def __init__(self, backbone, device,tracer, threshold = None):
            super().__init__()
            self.backbone = backbone.to(device)
            self.device = device
            self.tracer = tracer
            # threshold: scalar-like value used to compare against an image score.
            self.threshold = threshold
            # memory_bank: initially None; after fit, floating tensor (M, 1024),
            # where M is the number of patch representatives kept by coreset.
            self.register_buffer("memory_bank", None)
            self.feature_extractor= feature_extraction.create_feature_extractor(
                  model=self.backbone,
                  return_nodes={
                  "layer2": "layer2",
                  "layer3": "layer3"
            }
            ).to(device)


      def make_embeddings(self, features):
            # features: dict[str, Tensor] returned by feature_extractor.
            # For input (B, 3, H, W), Wide ResNet-50-2 normally produces:
            #   layer2: (B, 512, H2, W2), approximately H2=H/8 and W2=W/8
            #   layer3: (B, 1024, H3, W3), approximately H3=H/16 and W3=W/16
            # Output embds: floating Tensor (B, P, 1024), where P=H2*W2.
            unfolder = nn.Unfold(
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                  )
            # shape2=(B, 512, H2, W2); shape3=(B, 1024, H3, W3).
            shape2, shape3 = features["layer2"].shape, features["layer3"].shape
            # Unfold maps (B, C, Hx, Wx) -> (B, C*3*3, Hx*Wx), because
            # kernel=3, stride=1, and padding=1 preserve the patch grid size.
            # features_2: (B, 4608, H2*W2); features_3: (B, 9216, H3*W3).
            features_2, features_3= unfolder(features["layer2"]), unfolder(features["layer3"])

            # features_3: (B, 9216, H3, W3).
            features_3 = features_3.unflatten(-1, shape3[-2:])

            # interpolating layer3 features to match layer2 features
            # features_3: (B, 9216, H2, W2).
            features_3 = F.interpolate(features_3, 
                                          size = shape2[-2:],
                                          mode='bilinear', 
                                          align_corners=False)

            # features_3: (B, 9216, H2*W2).
            features_3 = features_3.flatten(start_dim =2)

            assert features_2.shape[-1] == features_3.shape[-1], \
            f"Patch count mismatch: {features_2.shape[-1]} != {features_3.shape[-1]}"
            # Both now use the patch-first layout (B, P, patch_embedding_dim):
            # features_2=(B, P, 4608), features_3=(B, P, 9216), P=H2*W2.
            features_2, features_3= features_2.transpose(1,2), features_3.transpose(1,2)
            # B: batch size; P: patches per image; emb2=4608; emb3=9216.
            B,P, emb2 = features_2.shape
            emb3 = features_3.shape[2]
            # Treat every patch as an independent 1D signal:
            # features_2=(B*P, 1, emb2); features_3=(B*P, 1, emb3).
            features_2, features_3 = features_2.reshape(B*P, 1, emb2), features_3.reshape(B*P, 1, emb3)

            # features_2 and features_3: (B*P, 1, 1024).
            features_2, features_3= F.adaptive_avg_pool1d(features_2, output_size=1024), \
                                    F.adaptive_avg_pool1d(features_3, output_size=1024)

            # features_2 and features_3: (B*P, 1024).
            features_2, features_3 = features_2.squeeze(1), features_3.squeeze(1)

            # embds: (B*P, 2, 1024), with one row per backbone layer.
            embds = torch.stack([features_2, features_3], dim=1)
            # embds: (B*P, 1, 2048).
            embds = embds.reshape(B*P, 1, 2048)
            # embds: (B*P, 1, 1024).
            embds = F.adaptive_avg_pool1d(embds, output_size=1024)
            # embds: (B*P, 1024).
            embds = embds.squeeze(1)
            # embds: (B, P, 1024).
            embds = embds.reshape(B,P,1024)

            return embds

      def coreset(self, embds, ratio ):
            # embds: floating Tensor (M_all, D), normally D=1024.
            # Output reprs is intended to contain about ratio*M_all rows.
            embds_idx = list(range(embds.shape[0]))
            # initial_rep_idx: Python int in [0, M_all).
            initial_rep_idx = random.choice(embds_idx)
            # reprs: (1, D).
            reprs = embds[initial_rep_idx].unsqueeze(0)

            # reprs[-1]: (D,). torch.cdist requires each input to have at least
            # two dimensions, with layouts (..., P, D) and (..., R, D).
            dist = torch.cdist(embds, reprs[-1].unsqueeze(0), p=2)

            # n: Python int, requested number of representatives.
            n = int(embds.shape[0] * ratio)

            with self.tracer.start_as_current_span("coreset selection") as span:
                  span.set_attribute("memory_bank.size", embds.shape[0])
                  span.set_attribute("coreset.ratio", ratio)
                  span.set_attribute("embedding.dimension", embds.shape[1])
                  span.set_attribute("device", str(embds.device))

                  for _ in tqdm(range(n-1), desc="Building coreset"):
                        # embds: (M_all, D); reprs[-1]: (D,).
                        # Profiling code
                        new_dist = torch.cdist(embds, reprs[-1].unsqueeze(0), p=2)


                        # Profiling code
                        # dist/new_dist would need matching shapes for elementwise minimum.
                        dist = torch.minimum(dist, new_dist)
                        # max_idx: scalar integer Tensor indexing the farthest embedding.
                        max_idx = dist.argmax()

                        # embds[max_idx] has shape (D,); unsqueeze(1) makes (D, 1).
                        new_repr = embds[max_idx].unsqueeze(0)

                        # reprs has shape (representatives, D), while new_repr is (D, 1).
                        reprs = torch.cat((reprs, new_repr), dim = 0)

            return reprs


      def embd_score(self, embds):
            """
            Comparing the embds with the embds in the memory bank
            """
            # embds: (..., P, D), commonly (B, P, 1024).
            # memory_bank: (M, D), commonly (M, 1024).
            # distances: (..., P, M), one distance per patch/representative pair.
            distances = torch.cdist(
                  x1= embds, 
                  x2 = self.memory_bank,
                  p=2
            )

            # patch_scores: (..., P), minimum memory-bank distance per patch.
            patch_scores  = torch.amin(distances, dim= -1)

            return patch_scores

      def batch_embds(self, dataloader):            
            # dataloader yields dict-like batches with:
            # batch["image"]: floating Tensor (B_i, 3, H, W).
            embd_batches = []
            with torch.inference_mode():
                  for batch in dataloader:
                        # images: (B_i, 3, H, W), on self.device.
                        images = batch["image"].to(self.device)
                        # features: {"layer2": (B_i, 512, H2, W2),
                        #            "layer3": (B_i, 1024, H3, W3)}.
                        features = self.feature_extractor(images)
                        # embds: (B_i, P, 1024), P=H2*W2.
                        embds = self.make_embeddings(features)
                        # Each stored tensor is (B_i, P, 1024), on CPU.
                        embd_batches.append(embds.detach().cpu())
            
            # embd_batches: (N, P, 1024), N=sum of all batch sizes.
            embd_batches = torch.cat(embd_batches, dim=0)
            return embd_batches.to(self.device)


      def calibrate_score(self,val_dataloader):
            # embd_batches: (N_val, P, 1024).
            embd_batches = self.batch_embds(val_dataloader)
            # patches_score: (N_val, P).
            patches_score = self.embd_score(embd_batches)
            # images_score: (N_val,), maximum patch score for each image.
            images_score = torch.amax(patches_score, dim=-1)
            # p99: scalar Tensor containing the 99th percentile image score.
            p99 = torch.quantile(images_score, 0.99 )
            self.threshold = p99





      def fit(self, train_dataloader, ratio):
            # embd_batches: (N, P, D), normally D=1024.
            embd_batches = self.batch_embds(train_dataloader)
            N,P,D = embd_batches.shape
            # embd_batches: (N*P, D), pooling every training-image patch.
            embd_batches =  embd_batches.reshape(N*P, D)
            # memory_bank: intended shape (M, D), where M is the coreset size.
            self.memory_bank = self.coreset(embd_batches, ratio).to(self.device)


      def predict(self, images):
            # batch is currently passed directly to make_embeddings, so its
            # expected layout is a feature dict with layer2/layer3 tensors.
            # embds: (B, P, 1024).
            features = self.feature_extractor(images)
            embds = self.make_embeddings(features)

            # patch_score: (B, P).
            patches_score = self.embd_score(embds)
            # image_score is intended to reduce the P dimension to shape (B,).
            image_score = patches_score.amax( dim=-1)


            predictions = image_score > self.threshold




            ## Placeholder

            return {
                  "patches_score": patches_score,
                  "image_score": image_score , 
                  "predictions": predictions.int()   
            }

      def evaluation(self, test_dataloader):
            avg_penality = 0 
            n = 0 
            with torch.inference_mode():
                  for batch in test_dataloader:
                        images = batch['image'].to(self.device)
                        predict_info = self.predict(images)
                        predicted_labels = predict_info['predictions']
                        labels = batch['label'].to(self.device)
                        avg_penality += (labels != predicted_labels).sum().item()
                        n += images.shape[0]
                  assert n!=0
                  avg_penality /= n

            

            return avg_penality

            



                  




            


            
                  





