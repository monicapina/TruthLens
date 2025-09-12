# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMultiPositiveInfoNCE(nn.Module):
    """InfoNCE multipositivo con pesos por par (0..1)."""
    def __init__(self, temperature: float = 0.07, eps: float = 1e-6):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0/temperature))
        self.eps = eps

    def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor, pos_weights: torch.Tensor) -> torch.Tensor:
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        logits = self.logit_scale.exp() * img_feats @ txt_feats.t()  # (B,T)
        denom = torch.logsumexp(logits, dim=1)                       # (B,)
        logw = torch.log(pos_weights.clamp(min=self.eps))            # (-inf para 0)
        pos_logits = logits + logw
        pos_lse = torch.logsumexp(pos_logits, dim=1)                 # (B,)
        return -(pos_lse - denom).mean()

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negative, dim=-1)
        d_ap = 1 - (a*p).sum(dim=-1)
        d_an = 1 - (a*n).sum(dim=-1)
        return F.relu(d_ap - d_an + self.margin).mean()
