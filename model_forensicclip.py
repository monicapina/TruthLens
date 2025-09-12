# model_forensicclip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class ForensicCLIP(nn.Module):
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k"):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        d = self.clip.text_projection.shape[1]
        self.c_real = nn.Parameter(torch.randn(d))
        self.c_fake = nn.Parameter(torch.randn(d))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.clip.encode_image(images)
        return F.normalize(feats, dim=-1)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        feats = self.clip.encode_text(text_tokens)
        return F.normalize(feats, dim=-1)

    def class_logits(self, img_feats: torch.Tensor) -> torch.Tensor:
        C = torch.stack([
            F.normalize(self.c_real, dim=0),
            F.normalize(self.c_fake, dim=0)
        ], dim=0)  # (2, D)
        return img_feats @ C.t()
