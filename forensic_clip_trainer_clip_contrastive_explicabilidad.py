# =============================
# Project: ForensicCLIP Trainer
# CLIP finetune (image/text encoders + projections) with:
#  - Multi-positive InfoNCE (image ↔ attribute prompts + captions)
#  - Binary classification via class prototypes (no text needed at test)
#  - Optional temporal Triplet loss using provided frame triplets
#  - Attribute regression vs attr_targets (soft supervision)
#  - Inference that returns verdict, confidence, and top-K forensic reasons
# =============================
# Folder layout (suggested):
# forensicclip/
#   prompts_bank.py
#   dataset_json.py
#   model_forensicclip.py
#   losses.py
#   train.py
#   infer.py
#   utils.py
#
# Usage (examples):
#   python train.py --json /path/to/truthlens_vlm.json --outdir runs/run1
#   python infer.py --ckpt runs/run1/best.pt --image /path/img.jpg --topk 5
#   python infer.py --ckpt runs/run1/best.pt --video /path/video.mp4 --num_frames 16

# =============================
# File: prompts_bank.py
# =============================
from typing import Dict, List

ATTRIBUTES: Dict[str, Dict[str, List[str]]] = {
    "lighting_consistency": {
        "suspicious": [
            "lighting inconsistency between face and background",
            "inconsistent shadow direction on the face",
            "mismatched highlights and shadows across face and scene",
        ],
        "realistic": [
            "consistent lighting on face and background",
            "shadows and highlights align with scene lighting",
        ],
    },
    "skin_texture": {
        "suspicious": [
            "unnatural skin texture",
            "waxy or overly smooth facial texture",
            "texture inconsistency between facial regions",
        ],
        "realistic": [
            "natural skin texture with fine detail",
            "consistent facial texture across regions",
        ],
    },
    "eye_blinking": {
        "suspicious": [
            "unnatural eye blinking pattern",
            "irregular or absent eye blinks",
        ],
        "realistic": [
            "natural and regular eye blinking",
        ],
    },
    "lip_motion_consistency": {
        "suspicious": [
            "lip motion inconsistent with speech",
            "asynchronous lip movements",
        ],
        "realistic": [
            "lip motion consistent with speech",
        ],
    },
    "contour_blending": {
        "suspicious": [
            "blurred or inconsistent face contour blending",
            "halo artifacts near facial boundary",
        ],
        "realistic": [
            "clean and consistent facial boundary",
        ],
    },
    "pose_3d": {
        "suspicious": [
            "inconsistent 3D head pose with background",
            "impossible head pose transitions",
        ],
        "realistic": [
            "head pose consistent with scene and camera",
        ],
    },
    "temporal_artifacts": {
        "suspicious": [
            "temporal flickering across frames",
            "inconsistent details between consecutive frames",
        ],
        "realistic": [
            "temporal consistency across frames",
        ],
    },
    "face_background_separation": {
        "suspicious": [
            "unnatural separation between face and background",
            "depth or matting inconsistency around face",
        ],
        "realistic": [
            "natural separation between face and background",
        ],
    },
    "accessories_consistency": {
        "suspicious": [
            "inconsistent rendering of accessories such as hats or glasses",
        ],
        "realistic": [
            "accessories rendered consistently with lighting and pose",
        ],
    },
    "expressive_dynamics": {
        "suspicious": [
            "unnatural facial expression dynamics",
            "rigid or mismatched expressions over time",
        ],
        "realistic": [
            "natural facial expression dynamics",
        ],
    },
}

CLASS_PROMPTS = {
    "real": [
        "this is a real, unmanipulated human face image",
        "authentic human face without manipulation",
    ],
    "deepfake": [
        "this is a manipulated deepfake human face image",
        "human face generated or altered by deepfake techniques",
    ],
}

# =============================
# File: dataset_json.py
# =============================
import json, os, random
from typing import Any, Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from prompts_bank import ATTRIBUTES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _label_to_int(lbl: str) -> int:
    return 1 if lbl.lower() in {"deepfake", "fake", "manipulated"} else 0

class TruthLensVLMDataset(Dataset):
    def __init__(self, json_path: str, split: str = "train", val_ratio: float = 0.1,
                 image_size: int = 224, use_frame_level: bool = True, seed: int = 17):
        super().__init__()
        self.json_path = json_path
        self.split = split
        self.use_frame = use_frame_level
        random.seed(seed)
        with open(json_path, "r") as f:
            data = json.load(f)
        # build items
        videos = data["videos"]
        # simple split by video_id
        vid_ids = [v["video_id"] for v in videos]
        random.shuffle(vid_ids)
        n_val = max(1, int(len(vid_ids) * val_ratio))
        val_set = set(vid_ids[:n_val])
        def belongs(v):
            return (v["video_id"] in val_set) if split=="val" else (v["video_id"] not in val_set)
        self.items = []
        for v in videos:
            if not belongs(v):
                continue
            v_label = _label_to_int(v.get("expected_label", v.get("global_prediction", "real")))
            attr_targets = v.get("meta", {}).get("attr_targets", {})
            for fr in v.get("frames", []):
                img_path = fr.get("image_path")
                if not img_path or not os.path.splitext(img_path)[1].lower() in IMG_EXTS:
                    continue
                y = _label_to_int(fr.get("expected_label", v_label))
                caption = fr.get("caption")
                triplets = fr.get("triplet_paths", [])
                self.items.append({
                    "image": img_path,
                    "label": y,
                    "caption": caption,
                    "attr_targets": attr_targets,
                    "triplets": triplets,
                })
        self.tfm = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.items)

    def _load_img(self, path: str):
        img = Image.open(path).convert("RGB")
        return self.tfm(img)

    def _make_attr_positive_prompts(self, attr_targets: Dict[str, float]) -> Dict[str, str]:
        pos = {}
        for a, t in attr_targets.items():
            if a not in ATTRIBUTES:
                continue
            state = "suspicious" if t >= 0.5 else "realistic"
            cand = ATTRIBUTES[a][state]
            if len(cand) == 0:
                continue
            pos[a] = random.choice(cand)
        return pos

    def __getitem__(self, idx):
        it = self.items[idx]
        img = self._load_img(it["image"])
        y = torch.tensor(it["label"], dtype=torch.long)
        # positives: attribute prompts + optional caption
        attr_pos = self._make_attr_positive_prompts(it["attr_targets"])  # {attr: prompt}
        caption = it["caption"]  # may be None
        triplets = it["triplets"] or []
        triplet_imgs = None
        if len(triplets) >= 2:
            # take first and last as pos/neg candidate (anchor is current image)
            try:
                pos_img = self._load_img(triplets[len(triplets)//2])
                neg_img = self._load_img(triplets[-1])
                triplet_imgs = (img, pos_img, neg_img)
            except Exception:
                triplet_imgs = None
        sample = {
            "image": img,
            "label": y,
            "attr_pos": attr_pos,   # dict attr->prompt (one per attr)
            "caption": caption,     # optional
            "triplet": triplet_imgs # optional (anchor,pos,neg)
        }
        return sample

# =============================
# File: losses.py
# =============================
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPositiveInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0/temperature))

    def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor,
                pos_mask: torch.Tensor) -> torch.Tensor:
        """
        Legacy boolean-mask version kept for backwards-compat.
        img_feats: (B, D)
        txt_feats: (T, D)
        pos_mask:  (B, T) boolean, True where (img i, text j) is positive
        """
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        logits = self.logit_scale.exp() * img_feats @ txt_feats.t()  # (B, T)
        denom = torch.logsumexp(logits, dim=1)  # (B,)
        pos_logits = logits.masked_fill(~pos_mask, float('-inf'))
        pos_lse = torch.logsumexp(pos_logits, dim=1)
        loss = -(pos_lse - denom).mean()
        return loss

class WeightedMultiPositiveInfoNCE(nn.Module):
    """Multi-positive InfoNCE with per-pair weights.
    If weights[i,j]==0 → negative; >0 → positive with strength proportional to weight.
    """
    def __init__(self, temperature: float = 0.07, eps: float = 1e-6):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0/temperature))
        self.eps = eps

    def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor,
                pos_weights: torch.Tensor) -> torch.Tensor:
        """
        img_feats: (B, D)
        txt_feats: (T, D)
        pos_weights: (B, T) float in [0, 1] ideally
        """
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        logits = self.logit_scale.exp() * img_feats @ txt_feats.t()  # (B, T)
        denom = torch.logsumexp(logits, dim=1)  # (B,)
        # log-sum over positives with weights: log sum_j w_ij * exp(logit_ij)
        # = logsumexp(logit_ij + log w_ij)
        logw = torch.log(pos_weights.clamp(min=self.eps))  # (-inf for 0)
        pos_logits = logits + logw
        pos_lse = torch.logsumexp(pos_logits, dim=1)
        loss = -(pos_lse - denom).mean()
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        # cosine distance
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negative, dim=-1)
        d_ap = 1 - (a * p).sum(dim=-1)
        d_an = 1 - (a * n).sum(dim=-1)
        return F.relu(d_ap - d_an + self.margin).mean()
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        # cosine distance
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negative, dim=-1)
        d_ap = 1 - (a * p).sum(dim=-1)
        d_an = 1 - (a * n).sum(dim=-1)
        return F.relu(d_ap - d_an + self.margin).mean()

# =============================
# File: model_forensicclip.py
# =============================
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import Dict, List, Tuple

class ForensicCLIP(nn.Module):
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k"):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        # Learned class prototypes in text/image embed space dimension
        d = self.clip.text_projection.shape[1]
        self.c_real = nn.Parameter(torch.randn(d))
        self.c_fake = nn.Parameter(torch.randn(d))
        # Optional: small MLP projection on image features (keeps dim)
        self.img_proj = nn.Identity()
        self.txt_proj = nn.Identity()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.clip.encode_image(images)
        feats = F.normalize(feats, dim=-1)
        return self.img_proj(feats)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        feats = self.clip.encode_text(text_tokens)
        feats = F.normalize(feats, dim=-1)
        return self.txt_proj(feats)

    def class_logits(self, img_feats: torch.Tensor) -> torch.Tensor:
        C = torch.stack([
            F.normalize(self.c_real, dim=0),
            F.normalize(self.c_fake, dim=0)
        ], dim=0)  # (2, D)
        return img_feats @ C.t()  # (B,2)

# =============================
# File: utils.py
# =============================
import torch, os
import open_clip
from typing import List, Dict, Tuple, Optional
from prompts_bank import ATTRIBUTES, CLASS_PROMPTS

class PromptBank:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.attr_index = []  # list of (attr, state, text)
        texts = []
        for a, d in ATTRIBUTES.items():
            for state, lst in d.items():
                for t in lst:
                    self.attr_index.append((a, state, t))
                    texts.append(t)
        # also include class prompts for initialization (not used as attributes)
        self.class_index = []
        for cls, lst in CLASS_PROMPTS.items():
            for t in lst:
                self.class_index.append((cls, t))
        self.attr_tokens = self.tokenizer(texts)
        self.class_tokens = self.tokenizer([t for _, t in self.class_index])

    def build_pos_mask(self, batch_attr_pos: List[Dict[str, str]]) -> torch.Tensor:
        """Return (B, T_attr) boolean mask: True if text j is a positive for image i."""
        B = len(batch_attr_pos)
        T = len(self.attr_index)
        mask = torch.zeros((B, T), dtype=torch.bool)
        # map text -> idx
        text2idx = {}
        for j, (_, _, t) in enumerate(self.attr_index):
            text2idx.setdefault(t, []).append(j)
        for i, ap in enumerate(batch_attr_pos):
            for _, txt in ap.items():
                if txt in text2idx:
                    for j in text2idx[txt]:
                        mask[i, j] = True
        # ensure at least one positive per row (fallback: none → random true)
        for i in range(B):
            if not mask[i].any():
                mask[i, torch.randint(0, T, (1,))] = True
        return mask

    # -----------------------------
    # Fuzzy matching: batched version returning per-sample indices+scores
    # -----------------------------
    def fuzzy_attr_from_texts_batched(self, model, device, free_texts: List[Optional[str]], top_m: int = 2,
                                      thr: float = 0.18):
        """Return list (len=B) with per-sample list of (idx, score) where idx indexes attr_index.
        Scores are cosine similarities in [ -1, 1 ]. Only >= thr are returned.
        """
        # Build queries only for non-empty
        idx_map = {i: t for i, t in enumerate(free_texts) if t}
        if not idx_map:
            return [[] for _ in free_texts]
        tok = self.tokenizer([idx_map[i] for i in idx_map]).to(device)
        with torch.no_grad():
            q = model.encode_text(tok)  # (Nq, D)
            bank = model.encode_text(self.attr_tokens.to(device))  # (T, D)
            S = q @ bank.t()  # (Nq, T)
        per = [[] for _ in free_texts]
        for k, (row_i, text) in enumerate(idx_map.items()):
            sims, idxs = torch.topk(S[k], k=min(top_m, S.shape[1]))
            for s, j in zip(sims.tolist(), idxs.tolist()):
                if s >= thr:
                    per[row_i].append((j, float(s)))
        return per

    # -----------------------------
    # NEW: fuzzy match free-form texts (e.g., VLM captions) to attribute prompts
    # using CLIP text-encoder cosine similarity.
    # -----------------------------
    def fuzzy_attr_from_texts(self, model, device, free_texts: List[str], top_m: int = 2,
                               thr: float = 0.18) -> List[Tuple[str, str, str, float]]:
        """
        Map arbitrary short texts to the closest attribute prompts.
        Returns a list of tuples (attr, state, matched_prompt, score) for top matches above thr.
        - top_m: how many matches to return
        - thr: minimum cosine similarity to accept a match (0..1). 0.18–0.25 works well.
        """
        if not free_texts:
            return []
        tok = self.tokenizer(free_texts).to(device)
        with torch.no_grad():
            q = model.encode_text(tok)  # (Nq, D)
            bank = model.encode_text(self.attr_tokens.to(device))  # (T, D)
            S = q @ bank.t()  # (Nq, T), feats are normalized in encode_text
        results = []
        for i in range(S.size(0)):
            sims, idxs = torch.topk(S[i], k=min(top_m, S.size(1)))
            for s, j in zip(sims.tolist(), idxs.tolist()):
                if s < thr:
                    continue
                a, state, text = self.attr_index[j]
                results.append((a, state, text, float(s)))
        # deduplicate by (attr,state,text), keep highest score
        best = {}
        for a, st, tx, sc in results:
            key = (a, st, tx)
            if key not in best or sc > best[key]:
                best[key] = sc
        # sort by score
        out = [(a, st, tx, sc) for (a, st, tx), sc in best.items()]
        out.sort(key=lambda x: x[3], reverse=True)
        return out[:top_m]

# =============================
# File: train.py
# =============================
import os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import open_clip
from dataset_json import TruthLensVLMDataset
from model_forensicclip import ForensicCLIP
from losses import MultiPositiveInfoNCE, WeightedMultiPositiveInfoNCE, TripletLoss
from utils import PromptBank


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data
    ds_tr = TruthLensVLMDataset(args.json, split='train', image_size=args.image_size)
    ds_va = TruthLensVLMDataset(args.json, split='val', image_size=args.image_size)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model & tokenizer
    model = ForensicCLIP(args.model, args.pretrained).to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    bank = PromptBank(tokenizer)

    # Pre-encode class prompts to initialize prototypes (optional)
    with torch.no_grad():
        txt_feats = model.encode_text(bank.class_tokens.to(device))  # (Nc, D)
        real_idx = [i for i,(c,_) in enumerate(bank.class_index) if c=="real"]
        fake_idx = [i for i,(c,_) in enumerate(bank.class_index) if c=="deepfake"]
        model.c_real.copy_(txt_feats[real_idx].mean(dim=0).detach())
        model.c_fake.copy_(txt_feats[fake_idx].mean(dim=0).detach())

    # Losses
    loss_con = WeightedMultiPositiveInfoNCE(args.temperature).to(device)
    loss_tri = TripletLoss(args.triplet_margin).to(device) if args.use_triplet else None
    loss_ce  = nn.CrossEntropyLoss()

    # Optim
    params = list(model.parameters()) + list(loss_con.parameters())
    optim_all = optim.AdamW(params, lr=args.lr, weight_decay=0.2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_all, T_max=args.epochs*len(dl_tr))

    best_val = 0.0
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, 'best.pt')

    for epoch in range(1, args.epochs+1):
        model.train()
        tot = 0.0
        for batch in dl_tr:
            optim_all.zero_grad()
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            # Encode image
            img_feats = model.encode_image(imgs)  # (B,D)

            # -----------------------------
            # Build text batch (attributes + fuzzy matches from caption)
            batch_attr_pos = batch['attr_pos']  # dict-based positives from attr_targets
            pos_mask = bank.build_pos_mask(batch_attr_pos).to(device)  # (B,T) bool

            # Build base weights from mask (1.0 for dictionary positives, else 0)
            pos_w = pos_mask.float()

            # Add fuzzy-caption positives with weights proportional to cosine similarity
            cap_texts = [c for c in batch['caption']]
            fuzzy = bank.fuzzy_attr_from_texts_batched(model, device, cap_texts, top_m=args.fuzzy_topm, thr=args.fuzzy_thr)
            # normalize similarity -> weight in [0,1]: w = alpha * ( (s - thr) / (1 - thr) )
            alpha = args.caption_weight
            for i, hits in enumerate(fuzzy):
                for j, s in hits:
                    w = alpha * max(0.0, (s - args.fuzzy_thr) / (1.0 - args.fuzzy_thr + 1e-6))
                    pos_w[i, j] = torch.clamp(pos_w[i, j] + w, max=1.0)

            attr_tokens = bank.attr_tokens.to(device)   # (T_attr, L)
            txt_feats = model.encode_text(attr_tokens)  # (T_attr,D)
            L_con = loss_con(img_feats, txt_feats, pos_w)

            # Classification (no text needed)
            logits = model.class_logits(img_feats)
            L_cls = loss_ce(logits, labels)

            # Triplet (optional)
            L_tri = torch.tensor(0.0, device=device)
            if loss_tri is not None:
                trips = batch['triplet']
                if trips is not None:
                    anchors, poss, negs = [], [], []
                    for t in trips:
                        if t is not None:
                            a,p,n = t
                            anchors.append(a)
                            poss.append(p)
                            negs.append(n)
                    if len(anchors) > 0:
                        A = torch.stack(anchors).to(device)
                        P = torch.stack(poss).to(device)
                        N = torch.stack(negs).to(device)
                        fA = model.encode_image(A)
                        fP = model.encode_image(P)
                        fN = model.encode_image(N)
                        L_tri = loss_tri(fA, fP, fN)

            loss = args.w_con*L_con + args.w_cls*L_cls + args.w_tri*L_tri
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_all.step()
            scheduler.step()
            tot += loss.item()
        # simple validation (accuracy on val)
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for batch in dl_va:
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)
                feats = model.encode_image(imgs)
                logits = model.class_logits(feats)
                pred = logits.argmax(dim=1)
                correct += (pred==labels).sum().item()
                total += labels.numel()
        acc = correct/ max(1,total)
        print(f"Epoch {epoch}: train_loss={tot/len(dl_tr):.4f} val_acc={acc:.4f}")
        if acc > best_val:
            best_val = acc
            torch.save({
                'model': model.state_dict(),
                'args': vars(args)
            }, ckpt_path)
    print(f"Best val acc: {best_val:.4f} (saved to {ckpt_path})")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=str, required=True)
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--model', type=str, default='ViT-B-16')
    ap.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--use_triplet', action='store_true')
    ap.add_argument('--triplet_margin', type=float, default=0.2)
    ap.add_argument('--w_con', type=float, default=1.0)
    ap.add_argument('--w_cls', type=float, default=1.0)
    ap.add_argument('--w_tri', type=float, default=0.2)
    # Fuzzy caption integration
    ap.add_argument('--fuzzy_thr', type=float, default=0.2)
    ap.add_argument('--fuzzy_topm', type=int, default=2)
    ap.add_argument('--caption_weight', type=float, default=0.5, help='weight multiplier for caption-based positives (0..1)')
    args = ap.parse_args()
    train(args)
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=str, required=True)
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--model', type=str, default='ViT-B-16')
    ap.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--use_triplet', action='store_true')
    ap.add_argument('--triplet_margin', type=float, default=0.2)
    ap.add_argument('--w_con', type=float, default=1.0)
    ap.add_argument('--w_cls', type=float, default=1.0)
    ap.add_argument('--w_tri', type=float, default=0.2)
    ap.add_argument('--w_attr', type=float, default=0.2)
    args = ap.parse_args()
    train(args)

# =============================
# File: infer.py
# =============================
import argparse, torch, os
import open_clip
import numpy as np
from PIL import Image
from torchvision import transforms
from model_forensicclip import ForensicCLIP
from utils import PromptBank

IMN = (0.48145466, 0.4578275, 0.40821073)
IMS = (0.26862954, 0.26130258, 0.27577711)

def preprocess(image_size=224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMN, IMS),
    ])

@torch.no_grad()
def explain_image(model, bank, img_path, topk=5, device='cuda'):
    tfm = preprocess()
    img = Image.open(img_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)
    f = model.encode_image(x)  # (1,D)
    # verdict
    logits = model.class_logits(f)
    prob = torch.softmax(logits, dim=1)[0,1].item()
    verdict = 'deepfake' if prob>=0.5 else 'real'
    # attribute similarities
    txt_feats = model.encode_text(bank.attr_tokens.to(device))  # (T,D)
    sim = (f @ txt_feats.t()).squeeze(0)  # (T,)
    # gather topk by absolute similarity (or just highest)
    top_idx = torch.topk(sim, k=min(topk, sim.numel())).indices.tolist()
    reasons = []
    for j in top_idx:
        a, state, text = bank.attr_index[j]
        reasons.append({"attribute": a, "state": state, "prompt": text, "score": float(sim[j].item())})
    return verdict, float(prob), reasons

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--image', type=str, default=None)
    ap.add_argument('--video', type=str, default=None)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--model', type=str, default='ViT-B-16')
    ap.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ForensicCLIP(args.model, args.pretrained).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'])
    tokenizer = open_clip.get_tokenizer(args.model)
    bank = PromptBank(tokenizer)
    if args.image:
        v, p, reasons = explain_image(model, bank, args.image, args.topk, device)
        print({"verdict": v, "confidence": p, "top_reasons": reasons})
    elif args.video:
        # simple uniform sampling
        import cv2
        cap = cv2.VideoCapture(args.video)
        frames = []
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(0,n-1), num=16, dtype=int)
        tfm = preprocess()
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok: continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
            frames.append(tfm(img))
        cap.release()
        if len(frames)==0:
            print("No frames decoded")
            exit(0)
        X = torch.stack(frames).to(device)
        with torch.no_grad():
            F = model.encode_image(X)  # (T,D)
            logits = model.class_logits(F)
            probs = torch.softmax(logits, dim=1)[:,1]  # (T,)
            conf = float(probs.mean().item())
            verdict = 'deepfake' if conf>=0.5 else 'real'
            # reasons: average similarity
            txt_feats = model.encode_text(bank.attr_tokens.to(device))  # (Ttxt,D)
            sim = F @ txt_feats.t()  # (Tfrm, Ttxt)
            mean_sim = sim.mean(dim=0)  # (Ttxt,)
            top_idx = torch.topk(mean_sim, k=min(args.topk, mean_sim.numel())).indices.tolist()
            reasons = []
            for j in top_idx:
                a, state, text = bank.attr_index[j]
                reasons.append({"attribute": a, "state": state, "prompt": text, "score": float(mean_sim[j].item())})
            print({"verdict": verdict, "confidence": conf, "top_reasons": reasons})
    else:
        print("Provide --image or --video")
