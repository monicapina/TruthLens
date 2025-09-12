# utils.py
import torch
import open_clip
from typing import List, Dict, Optional
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
        self.class_index = []
        for cls, lst in CLASS_PROMPTS.items():
            for t in lst:
                self.class_index.append((cls, t))
        self.attr_tokens = self.tokenizer(texts)
        self.class_tokens = self.tokenizer([t for _, t in self.class_index])

    def build_pos_mask(self, batch_attr_pos: List[Dict[str, str]]) -> torch.Tensor:
        B = len(batch_attr_pos)
        T = len(self.attr_index)
        mask = torch.zeros((B, T), dtype=torch.bool)
        # text -> idx
        text2idx = {}
        for j, (_, _, t) in enumerate(self.attr_index):
            text2idx.setdefault(t, []).append(j)
        for i, ap in enumerate(batch_attr_pos):
            for _, txt in ap.items():
                if txt in text2idx:
                    for j in text2idx[txt]:
                        mask[i, j] = True
        for i in range(B):
            if not mask[i].any():
                mask[i, torch.randint(0, T, (1,))] = True
        return mask

    def fuzzy_attr_from_texts_batched(self, model, device, free_texts: List[Optional[str]], top_m: int = 2, thr: float = 0.18):
        idx_map = {i: t for i, t in enumerate(free_texts) if t}
        if not idx_map:
            return [[] for _ in free_texts]
        tok = self.tokenizer([idx_map[i] for i in idx_map]).to(device)
        with torch.no_grad():
            q = model.encode_text(tok)                           # (Nq, D)
            bank = model.encode_text(self.attr_tokens.to(device))# (T, D)
            S = q @ bank.t()                                     # (Nq, T)
        per = [[] for _ in free_texts]
        for k, (row_i, _) in enumerate(idx_map.items()):
            sims, idxs = torch.topk(S[k], k=min(top_m, S.shape[1]))
            for s, j in zip(sims.tolist(), idxs.tolist()):
                if s >= thr:
                    per[row_i].append((j, float(s)))
        return per
