# dataset_json.py
import os, json, random
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMN = (0.48145466, 0.4578275, 0.40821073)
IMS = (0.26862954, 0.26130258, 0.27577711)

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _collect_videos(json_paths: List[str]) -> List[Dict[str, Any]]:
    vids = []
    for p in json_paths:
        with open(p, "r") as f:
            d = json.load(f)
        vids.extend(d.get("videos", []))
    return vids

def _stratified_video_split(videos: List[Dict[str, Any]], val_ratio=0.1, seed=42) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    by_lab = {"real": [], "deepfake": []}
    for v in videos:
        lab = v.get("expected_label", "real")
        by_lab.setdefault(lab, []).append(v["video_id"])
    tr_ids, va_ids = [], []
    for lab, ids in by_lab.items():
        ids = list(set(ids))
        random.shuffle(ids)
        n_val = max(1, int(round(len(ids)*val_ratio))) if len(ids) > 0 else 0
        va_ids.extend(ids[:n_val])
        tr_ids.extend(ids[n_val:])
    return tr_ids, va_ids

class TruthLensVLMDataset(Dataset):
    def __init__(self, json_path, split='train', image_size=224, val_ratio=0.1, seed=42):
        json_paths = _ensure_list(json_path)
        videos = _collect_videos(json_paths)

        # split por video_id (estratificado por expected_label)
        tr_ids, va_ids = _stratified_video_split(videos, val_ratio=val_ratio, seed=seed)
        keep = set(tr_ids if split=='train' else va_ids)

        # index rápido de videos por id
        vid_by_id = {v["video_id"]: v for v in videos if v["video_id"] in keep}

        self.samples = []
        for vid in vid_by_id.values():
            vlab = 0 if vid.get("expected_label","real")=="real" else 1
            attr_targets = vid.get("meta", {}).get("attr_targets", {})
            for fr in vid.get("frames", []):
                rec = {
                    "image_path": fr["image_path"],
                    "label": vlab,
                    "caption": fr.get("caption", None),
                    "attr_pos": attr_targets,
                    "triplet_paths": fr.get("triplet_paths", None)
                }
                self.samples.append(rec)

        self.tfm = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMN, IMS)
        ])

        # métricas de sanity
        self.split = split
        self.video_ids = list(keep)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        img = self.tfm(img)
        label = torch.tensor(s["label"], dtype=torch.long)
        caption = s.get("caption", None)

        triplet = None
        tpaths = s.get("triplet_paths", None)
        if tpaths and len(tpaths) >= 3:
            try:
                a = self.tfm(Image.open(tpaths[0]).convert("RGB"))
                p = self.tfm(Image.open(tpaths[1]).convert("RGB"))
                n = self.tfm(Image.open(tpaths[2]).convert("RGB"))
                triplet = (a, p, n)
            except Exception:
                triplet = None

        return {
            "image": img,
            "label": label,
            "caption": caption,
            "attr_pos": s.get("attr_pos", {}),
            "triplet": triplet
        }
