import os, argparse, csv
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import open_clip

from dataset_json import TruthLensVLMDataset
from model_forensicclip import ForensicCLIP
from losses import WeightedMultiPositiveInfoNCE, TripletLoss
from utils import PromptBank

# sklearn opcional (para F1/AUC)
try:
    from sklearn.metrics import f1_score, roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# -----------------------------
# Collate que tolera None y strings/dicts
# -----------------------------
def collate_safe(batch):
    images  = torch.stack([b["image"] for b in batch], dim=0)   # (B,C,H,W)
    labels  = torch.stack([b["label"] for b in batch], dim=0)   # (B,)
    captions = [b.get("caption", None) for b in batch]          # List[str|None]
    attr_pos = [b.get("attr_pos", {}) for b in batch]           # List[dict]
    trips    = [b.get("triplet", None) for b in batch]          # List[(img,img,img)|None]
    return {"image": images, "label": labels, "caption": captions, "attr_pos": attr_pos, "triplet": trips}


def maybe_write_metrics_header(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            cols = ["epoch", "train_loss", "val_acc", "val_f1_macro", "val_auc", "best_thr_f1"]
            w.writerow(cols)


def append_metrics(csv_path, epoch, train_loss, val_acc, val_f1, val_auc, best_thr):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_acc:.6f}",
            "" if val_f1 is None else f"{val_f1:.6f}",
            "" if val_auc is None else f"{val_auc:.6f}",
            "" if best_thr is None else f"{best_thr:.3f}",
        ])


def load_checkpoint_if_any(args, model, optimizer, scheduler, device):
    start_epoch = 1
    best_val = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] Cargando checkpoint desde: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and optimizer is not None and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and scheduler is not None and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best_val" in ckpt:
            best_val = float(ckpt["best_val"])
        print(f"[Resume] Reanudando en epoch {start_epoch} | best_val previo={best_val:.4f}")
    return start_epoch, best_val


def save_checkpoint(path, epoch, model, optimizer, scheduler, args, best_val=None):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "args": vars(args),
    }
    if best_val is not None:
        payload["best_val"] = float(best_val)
    torch.save(payload, path)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        pt = torch.softmax(logits, dim=1)[torch.arange(target.size(0), device=logits.device), target]
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------
    # Data (split por vídeo lo maneja dataset_json)
    # -----------------------------
    ds_tr = TruthLensVLMDataset(args.json, split='train', image_size=args.image_size)
    ds_va = TruthLensVLMDataset(args.json, split='val',   image_size=args.image_size)

    # Sanity split
    if hasattr(ds_tr, "video_ids") and hasattr(ds_va, "video_ids"):
        inter = len(set(ds_tr.video_ids) & set(ds_va.video_ids))
        print(f"videos train: {len(ds_tr.video_ids)} | videos val: {len(ds_va.video_ids)} | intersección: {inter}")

    # -----------------------------
    # Balanceo de clases
    # -----------------------------
    # Contar clases en TRAIN (0=real, 1=deepfake)
    cnt = Counter([s["label"] for s in ds_tr.samples]) if hasattr(ds_tr, "samples") else Counter()
    n0, n1 = cnt.get(0, 1), cnt.get(1, 1)
    N = n0 + n1
    w0 = N / (2.0 * n0)
    w1 = N / (2.0 * n1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    print(f"[class weights] real={w0:.3f} deepfake={w1:.3f} (n0={n0}, n1={n1})")

    # Sampler balanceado (opcional)
    sampler = None
    if args.use_sampler:
        sample_weights = []
        for s in ds_tr.samples:
            lab = s["label"]
            sample_weights.append(1.0 / (n1 if lab == 1 else n0))
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(ds_tr),
            replacement=True
        )
        print("[sampler] WeightedRandomSampler activado.")

    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=True,
        collate_fn=collate_safe
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_safe
    )

    # -----------------------------
    # Modelo y tokenizer
    # -----------------------------
    model = ForensicCLIP(args.model, args.pretrained).to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    bank = PromptBank(tokenizer)

    # Inicializa prototipos con prompts de clase
    with torch.no_grad():
        txt_feats = model.encode_text(bank.class_tokens.to(device))  # (Nc, D)
        real_idx = [i for i, (c, _) in enumerate(bank.class_index) if c == "real"]
        fake_idx = [i for i, (c, _) in enumerate(bank.class_index) if c == "deepfake"]
        model.c_real.copy_(txt_feats[real_idx].mean(dim=0).detach())
        model.c_fake.copy_(txt_feats[fake_idx].mean(dim=0).detach())

    # -----------------------------
    # Pérdidas y optimizador
    # -----------------------------
    loss_con = WeightedMultiPositiveInfoNCE(args.temperature).to(device)
    if args.focal:
        loss_ce  = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
        print(f"[loss] FocalLoss(gamma={args.focal_gamma}) con class_weights")
    else:
        loss_ce  = nn.CrossEntropyLoss(weight=class_weights)
        print("[loss] CrossEntropy con class_weights")

    loss_tri = TripletLoss(args.triplet_margin).to(device) if args.use_triplet else None

    params = list(model.parameters()) + list(loss_con.parameters())
    optim_all = optim.AdamW(params, lr=args.lr, weight_decay=0.2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_all, T_max=max(1, args.epochs) * max(1, len(dl_tr)))

    # Output dir
    os.makedirs(args.outdir, exist_ok=True)
    best_ckpt_path = os.path.join(args.outdir, 'best.pt')
    metrics_csv = os.path.join(args.outdir, 'metrics.csv')
    maybe_write_metrics_header(metrics_csv)

    # Resume si procede
    start_epoch, best_val = load_checkpoint_if_any(args, model, optim_all, scheduler, device)

    # -----------------------------
    # Loop de entrenamiento
    # -----------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tot_loss = 0.0

        for batch in dl_tr:
            optim_all.zero_grad()

            imgs   = batch['image'].to(device)     # (B,C,H,W)
            labels = batch['label'].to(device)     # (B,)
            img_feats = model.encode_image(imgs)   # (B,D)

            # --- Positivos desde atributos (diccionario) ---
            batch_attr_pos = batch['attr_pos']                         # List[dict]
            pos_mask = bank.build_pos_mask(batch_attr_pos).to(device)  # (B,T_attr) bool
            pos_w = pos_mask.float()

            # --- Fuzzy: sumar pesos según similitud caption→atributo ---
            cap_texts = [c for c in batch['caption']]
            fuzzy = bank.fuzzy_attr_from_texts_batched(
                model, device, cap_texts, top_m=args.fuzzy_topm, thr=args.fuzzy_thr
            )
            alpha = args.caption_weight
            for i, hits in enumerate(fuzzy):
                for j, s in hits:
                    w = alpha * max(0.0, (s - args.fuzzy_thr) / (1.0 - args.fuzzy_thr + 1e-6))
                    pos_w[i, j] = torch.clamp(pos_w[i, j] + w, max=1.0)

            # Textos de atributos (estáticos)
            attr_tokens = bank.attr_tokens.to(device)
            txt_feats = model.encode_text(attr_tokens)             # (T_attr, D)

            # Contrastiva ponderada
            L_con = loss_con(img_feats, txt_feats, pos_w)

            # Clasificación binaria con prototipos
            logits = model.class_logits(img_feats)                 # (B,2)
            L_cls = loss_ce(logits, labels)

            # Triplet (opcional, si hay triplets válidos en el batch)
            L_tri = torch.tensor(0.0, device=device)
            if loss_tri is not None:
                trips = batch['triplet']
                if trips is not None:
                    anchors, poss, negs = [], [], []
                    for t in trips:
                        if t is not None:
                            a, p, n = t
                            anchors.append(a); poss.append(p); negs.append(n)
                    if len(anchors) > 0:
                        A = torch.stack(anchors).to(device)
                        P = torch.stack(poss).to(device)
                        N = torch.stack(negs).to(device)
                        fA = model.encode_image(A)
                        fP = model.encode_image(P)
                        fN = model.encode_image(N)
                        L_tri = loss_tri(fA, fP, fN)

            loss = args.w_con * L_con + args.w_cls * L_cls + args.w_tri * L_tri
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_all.step()
            scheduler.step()

            tot_loss += loss.item()

        # -----------------------------
        # Validación
        # -----------------------------
        model.eval()
        correct = 0; total = 0
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for batch in dl_va:
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)
                feats = model.encode_image(imgs)
                logits = model.class_logits(feats)
                probs = torch.softmax(logits, dim=1)[:, 1]  # prob de "deepfake"
                pred = logits.argmax(dim=1)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(pred.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

                correct += (pred == labels).sum().item()
                total += labels.numel()

        acc = correct / max(1, total)
        f1 = auc = None
        best_thr = None
        if _HAS_SKLEARN and len(set(all_labels)) > 1:
            try:
                f1 = f1_score(all_labels, all_preds, average='macro')
            except Exception:
                f1 = None
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except Exception:
                auc = None
            # Buscar umbral que maximiza F1 (macro) en validación
            import numpy as np
            thrs = np.linspace(0.1, 0.9, 17)
            best_f1, best_thr = -1.0, 0.5
            for thr in thrs:
                preds_thr = [1 if p >= thr else 0 for p in all_probs]
                f1_thr = f1_score(all_labels, preds_thr, average='macro')
                if f1_thr > best_f1:
                    best_f1, best_thr = f1_thr, thr

        # Logging por consola
        msg = f"Epoch {epoch}: train_loss={tot_loss/len(dl_tr):.4f} val_acc={acc:.4f}"
        if f1 is not None:
            msg += f" val_f1={f1:.4f}"
        if auc is not None:
            msg += f" val_auc={auc:.4f}"
        if best_thr is not None:
            msg += f" best_thr={best_thr:.2f}"
        print(msg)

        # Guardar métrica en CSV
        append_metrics(metrics_csv, epoch, tot_loss/len(dl_tr), acc, f1, auc, best_thr)

        # Best checkpoint (por val_acc)
        if acc > best_val:
            best_val = acc
            torch.save({'model': model.state_dict(), 'args': vars(args)}, best_ckpt_path)

        # Checkpoint periódico para reanudar
        if args.save_every > 0 and (epoch % args.save_every == 0):
            ckpt_path = os.path.join(args.outdir, f'checkpoint_epoch{epoch}.pt')
            save_checkpoint(ckpt_path, epoch, model, optim_all, scheduler, args, best_val=best_val)

    print(f"Best val acc: {best_val:.4f} (guardado en {best_ckpt_path})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=str, required=True, help="Ruta a 1 o varios JSONs (si tu dataset_json.py lo soporta).")
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--model', type=str, default='ViT-B-16')
    ap.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--temperature', type=float, default=0.07)

    # pérdidas
    ap.add_argument('--use_triplet', action='store_true')
    ap.add_argument('--triplet_margin', type=float, default=0.2)
    ap.add_argument('--w_con', type=float, default=1.0)
    ap.add_argument('--w_cls', type=float, default=1.0)
    ap.add_argument('--w_tri', type=float, default=0.2)

    # Fuzzy caption → atributos
    ap.add_argument('--fuzzy_thr', type=float, default=0.2)
    ap.add_argument('--fuzzy_topm', type=int, default=2)
    ap.add_argument('--caption_weight', type=float, default=0.5,
                    help='Peso (0..1) para positivos provenientes de captions fuzzeados.')

    # Balanceo
    ap.add_argument('--use_sampler', action='store_true', help='Activar sampler balanceado en el DataLoader de train.')
    ap.add_argument('--focal', action='store_true', help='Usar Focal Loss en lugar de CrossEntropy.')
    ap.add_argument('--focal_gamma', type=float, default=2.0)

    # Checkpoints / reanudar
    ap.add_argument('--save_every', type=int, default=1, help='Guardar checkpoint cada N épocas (0 = desactivar).')
    ap.add_argument('--resume', type=str, default='', help='Ruta a checkpoint_epochX.pt para reanudar.')

    args = ap.parse_args()
    train(args)
