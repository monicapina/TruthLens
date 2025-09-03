import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

import torch
from transformers import (
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
)

# =========================
# CONFIG
# =========================
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"  # checkpoint LLaVA-NeXT-Video (DPO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

ATTRIBUTES = {
    "lighting_inconsistent":
        "Are there inconsistencies in lighting direction or intensity across the face and background between frames?",
    "skin_texture_inconsistent":
        "Does the skin texture appear unnaturally smooth, plastic-like, or inconsistent between frames?",
    "blink_unnatural":
        "Are the eye blinks natural, with realistic frequency and variability, or do they look robotic or missing across frames?",
    "lip_sync_inconsistent":
        "Do the lip movements align naturally with expressions/speech across frames, or do they seem inconsistent or detached?",
    "contour_blending":
        "Do edges of face/hair/ears show unnatural blending, warping or flickering artifacts across frames?",
    "pose3d_inconsistent":
        "Does head pose transition naturally between frames, or are there abrupt jumps or unnatural 3D distortions?"
}

ATTR_PROMPTS = {
  # 1) Iluminación / sombras
  "lighting_consistency": [
    "Is facial lighting direction consistent across frames?",
    "Do face/neck shadows move coherently with head motion?",
    "Does facial lighting match background/environment lighting?"
  ],
  # 2) Highlights (specular)
  "specular_highlights": [
    "Are skin highlights stable and plausible across frames?",
    "Do highlights look unnaturally sharp or frozen in place?"
  ],
  # 3) Textura de piel
  "skin_texture": [
    "Is skin texture consistent (pores/wrinkles) over time?",
    "Does the skin appear overly smooth or plastic in any frames?",
    "Do pores or wrinkles pop in and out across frames?"
  ],
  # 4) Ojos / parpadeo
  "eye_blinking": [
    "Are blink frequency and duration natural across the clip?",
    "Do the eyes show flicker or rendering artifacts over frames?"
  ],
  # 5) Gaze / alineación ocular
  "eye_gaze_tracking": [
    "Do the eyes track consistently and remain well-aligned over time?"
  ],
  # 6) Boca / sincronía (sin audio: solo coherencia visual)
  "lip_motion_consistency": [
    "Are lip shapes and transitions smooth and consistent across frames?",
    "Do teeth/tongue look natural and stable when the mouth opens?"
  ],
  # 7) Contornos / blending
  "contour_blending": [
    "Are face/hair/ear boundaries clean without warping or flicker?",
    "Are there halo or edge-blending artifacts around the head?"
  ],
  # 8) Pose / geometría 3D
  "pose_3d": [
    "Is head pose evolution smooth without abrupt jumps or distortions?",
    "Are facial proportions stable across frames?"
  ],
  # 9) Coherencia temporal (artefactos globales)
  "temporal_artifacts": [
    "Do any facial parts flicker/warp/ghost between frames?",
    "Is there temporal smearing or displaced features in the sequence?"
  ],
  # 10) Accesorios (consistencia)
  "accessories_consistency": [
    "Do accessories (glasses/earrings) stay stable in alignment and shape?",
    "Do accessories ever disappear or deform unexpectedly?"
  ],
  # 11) Separación cara-fondo
  "face_background_separation": [
    "Is the face-background boundary consistent without color bleeding?"
  ],
  # 12) Dinámica expresiva / conducta
  "expressive_dynamics": [
    "Are facial expressions fluid and context-appropriate over time?",
    "Are there frozen expressions or overly stiff posture?"
  ],
}


INSTRUCTION = """You are a forensic video analyst specialized in deepfake detection.
You are given a sequence of consecutive frames from the same video.
Analyze ONLY the given attribute. Respond STRICTLY in JSON format:
{
  "label": "realistic" or "suspicious",
  "confidence": float between 0.0 and 1.0,
  "explanation": "short reason about the decision"
}"""

# =========================
# TOKEN HF
# =========================
def load_hf_token(path_json="token.json", path_txt="token.txt") -> Optional[str]:
    if os.path.exists(path_json):
        try:
            with open(path_json) as f:
                return json.load(f).get("hf_token")
        except Exception:
            pass
    if os.path.exists(path_txt):
        with open(path_txt) as f:
            return f.read().strip()
    # opcional: busca en HOME
    home_json = os.path.expanduser("~/.huggingface/token.json")
    home_txt  = os.path.expanduser("~/.huggingface/token.txt")
    if os.path.exists(home_json):
        try:
            with open(home_json) as f:
                return json.load(f).get("hf_token")
        except Exception:
            pass
    if os.path.exists(home_txt):
        with open(home_txt) as f:
            return f.read().strip()
    return None

# =========================
# VIDEO UTILS
# =========================
@dataclass
class WindowConf:
    window_size: int = 32      # nº frames por ventana
    stride: int = 16           # avance entre ventanas
    max_frames_model: int = 32 # máx frames que pasamos al modelo por ventana

def uniform_indices(total: int, num: int) -> List[int]:
    if num <= 0 or total <= 0:
        return []
    if num >= total:
        return list(range(total))
    return [int(round(i * (total - 1) / (num - 1))) for i in range(num)]

def frames_to_pil(frames_bgr: List[np.ndarray]) -> List[Image.Image]:
    out = []
    for fr in frames_bgr:
        fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        out.append(Image.fromarray(fr_rgb))
    return out

def generate_windows_from_video(video_path: str, wcfg: WindowConf) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    windows = []
    start = 0
    while start < total:
        end = min(start + wcfg.window_size, total)
        frames_bgr = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end):
            ok, fr = cap.read()
            if not ok:
                break
            frames_bgr.append(fr)

        if not frames_bgr:
            break

        idxs = uniform_indices(len(frames_bgr), min(len(frames_bgr), wcfg.max_frames_model))
        frames_sel = [frames_bgr[i] for i in idxs]
        frames_pil = frames_to_pil(frames_sel)

        t_start = start / fps
        t_end = (start + len(frames_bgr) - 1) / fps

        windows.append({
            "window_id": len(windows),
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "frames": frames_pil
        })

        if end == total:
            break
        start += wcfg.stride

    cap.release()
    return windows

# =========================
# LLaVA NEXT VIDEO
# =========================
def load_model_and_processor(model_id: str, hf_token: Optional[str]):
    processor = LlavaNextVideoProcessor.from_pretrained(model_id, token=hf_token)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto",
        token=hf_token
    ).eval()
    return model, processor

def safe_json_extract(text: str) -> Dict[str, Any]:
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            pass
    return {"raw_output": text}

def query_attribute(
        model, processor,
        frames_pil,               # lista de PIL.Image de la ventana
        attr_key: str,
        question: str,
        instruction: str,
        temperature: float = 0.0,
        max_new_tokens: int = 128
    ):
        # 1) Conversación con bloque SYSTEM y USER (texto + video)
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "A chat between a curious human and an artificial intelligence assistant. "
                            "The assistant gives helpful, detailed, and polite answers to the human's questions."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                    "text": f"{instruction}\n\nAttribute: {attr_key}\nQuestion: {question}"},
                    {"type": "video"}  # <<--- BLOQUE DE VÍDEO OBLIGATORIO
                ],
            },
        ]

        # 2) Aplica la plantilla de chat (inserta tokens especiales <video> / <image> / roles)
        chat_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,   # añade "assistant" para generar
            tokenize=False
        )

        # 3) Tokeniza y prepara batch=1 (listas) + padding
        inputs = processor(
            text=[chat_text],            # batch de 1
            videos=[frames_pil],         # batch de 1 (lista de frames PIL)
            padding=True,
            return_tensors="pt"
        )

        # 4) Mueve a dispositivo
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # 5) Generación (si temperature>0 activamos sampling)
        do_sample = temperature > 0.0
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=(temperature if do_sample else None)
            )

        # 6) Decodifica y parsea a JSON robusto
        text_out = processor.decode(output[0], skip_special_tokens=True)
        s, e = text_out.find("{"), text_out.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                parsed = json.loads(text_out[s:e+1])
            except Exception:
                parsed = {"raw_output": text_out}
        else:
            parsed = {"raw_output": text_out}

        # Normalización ligera de la etiqueta
        if "label" in parsed:
            lbl = str(parsed["label"]).lower()
            parsed["label"] = "suspicious" if "susp" in lbl else ("realistic" if "real" in lbl else lbl)

        # Limpia memoria (útil en bucles largos)
        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return parsed

import random

def analyze_window(model, processor, frames_pil, self_consistency_k=3, probes_per_attr=1):
    import numpy as np
    result = {}
    for attr_key, probes in ATTR_PROMPTS.items():
        # elige aleatoriamente N sondas (o usa round-robin si guardas un índice global)
        qs = random.sample(probes, k=min(probes_per_attr, len(probes)))

        runs_all = []
        for q in qs:
            runs = []
            for _ in range(self_consistency_k):
                out = query_attribute(
                    model, processor, frames_pil, attr_key, q, INSTRUCTION,
                    temperature=(0.7 if self_consistency_k > 1 else 0.0)
                )
                out["_probe"] = q  # guarda la sonda usada
                runs.append(out)
            runs_all.extend(runs)

        # agregación sobre todas las sondas elegidas (y sus repeticiones)
        confs = [float(r["confidence"]) for r in runs_all if isinstance(r.get("confidence"), (float, int))]
        labels = [str(r["label"]).lower() for r in runs_all if r.get("label")]
        exps = [r["explanation"] for r in runs_all if r.get("explanation")]

        agg = {}
        if labels:
            agg["label"] = max(set(labels), key=labels.count)
        if confs:
            agg["confidence_mean"] = float(np.mean(confs))
            agg["confidence_std"] = float(np.std(confs))
        if exps:
            # explicación más frecuente; si empate, la más corta
            agg["explanation"] = sorted(
                exps, key=lambda s: (exps.count(s), -len(str(s))), reverse=True
            )[0]
        agg["raw_runs"] = runs_all  # incluye _probe por cada salida
        result[attr_key] = agg
    return result

def summarize_video(windows_json: List[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np
    summary = {"attributes": {}}
    for attr_key in ATTRIBUTES.keys():
        confs, labels = [], []
        for w in windows_json:
            a = w["attributes"].get(attr_key, {})
            if isinstance(a.get("confidence_mean"), (float, int)):
                confs.append(float(a["confidence_mean"]))
            if a.get("label"):
                labels.append(a["label"])
        if confs:
            summary["attributes"][attr_key] = {
                "mean_confidence": float(np.mean(confs)),
                "std_confidence": float(np.std(confs)),
                "label_mode": (max(set(labels), key=labels.count) if labels else None)
            }
    votes = [v.get("label_mode") == "suspicious" for v in summary["attributes"].values() if v]
    if votes:
        summary["heuristic_decision"] = {
            "label": "deepfake" if sum(votes) > len(votes)/2 else "real",
            "suspicious_votes": int(sum(votes)),
            "total_considered": int(len(votes))
        }
    print(summary)
    return summary

# =========================
# PROCESO DE UN VÍDEO (reutilizable)
# =========================
def process_video(video_path: str, model, processor, win: int, stride: int, max_frames_model: int, k: int) -> Dict[str, Any]:
    wcfg = WindowConf(window_size=win, stride=stride, max_frames_model=max_frames_model)
    windows = generate_windows_from_video(video_path, wcfg)

    result = {
        "video_id": os.path.basename(video_path),
        "video_path": video_path,
        "windows": []
    }
    print("RESULT:",result["video_id"], "→", len(windows), "ventanas")
    for w in windows:
        attrs = analyze_window(model, processor, w["frames"], self_consistency_k=k)
        result["windows"].append({
            "window_id": w["window_id"],
            "t_start": w["t_start"],
            "t_end": w["t_end"],
            "attributes": attrs
        })
        print(w,result["windows"][-1]["attributes"])

    result["summary"] = summarize_video(result["windows"])
    return result

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    import argparse, glob

    parser = argparse.ArgumentParser(description="Deepfake attributes with LLaVA-NeXT-Video (HF).")
    parser.add_argument("--video", help="Ruta a un solo vídeo (mp4, etc.)")
    parser.add_argument("--data_dir", help="Carpeta con subcarpetas de vídeos")
    parser.add_argument("--out", default="dataset.json", help="Ruta del JSON de salida (si es carpeta, se usará dataset.json dentro)")
    parser.add_argument("--win", type=int, default=32, help="Frames por ventana")
    parser.add_argument("--stride", type=int, default=16, help="Stride entre ventanas")
    parser.add_argument("--max_frames_model", type=int, default=32, help="Máx frames pasados al modelo")
    parser.add_argument("--k", type=int, default=3, help="Self-consistency")
    parser.add_argument("--ext", default=".mp4", help="Extensión de vídeos a buscar")
    args = parser.parse_args()

    hf_token = load_hf_token()
    if not hf_token:
        raise RuntimeError("No se encontró token HF. Crea token.json {'hf_token':'...'} o token.txt con tu token.")

    print(f"[{time.strftime('%H:%M:%S')}] Cargando {MODEL_ID} ...")
    model, processor = load_model_and_processor(MODEL_ID, hf_token)

    # Normaliza salida: si 'out' es carpeta, crea dataset.json dentro
    out_path = args.out
    if not out_path.endswith(".json"):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "dataset.json")

    if args.video:
        # ----- Modo: un solo vídeo -----
        video_result = process_video(args.video, model, processor, args.win, args.stride, args.max_frames_model, args.k)
        dataset_json = {
            "dataset_name": "TruthLens-VLM",
            "model_id": MODEL_ID,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos": [video_result]
        }

    elif args.data_dir:
        videos = glob.glob(os.path.join(args.data_dir, "**", f"*{args.ext}"), recursive=True)
        print(f"[INFO] Encontrados {len(videos)} vídeos en {args.data_dir}")

        # si args.out es carpeta, úsala; si es archivo .json, lo convertimos en carpeta
        out_dir = args.out
        if out_dir.endswith(".json"):
            out_dir = os.path.splitext(out_dir)[0]
        os.makedirs(out_dir, exist_ok=True)

        manifest = {
            "dataset_name": "TruthLens-VLM",
            "model_id": MODEL_ID,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos": []   # aquí solo metemos metadatos, no todo el contenido
        }

        for vid in videos:
            try:
                res = process_video(vid, model, processor, args.win, args.stride, args.max_frames_model, args.k)

                # nombre de salida por vídeo
                rel = os.path.relpath(vid, args.data_dir)
                base = os.path.splitext(rel)[0].replace(os.sep, "_")
                out_json = os.path.join(out_dir, f"{base}.json")

                # guarda inmediatamente el resultado de este vídeo
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)

                # añade una entrada ligera al manifiesto
                manifest["videos"].append({
                    "video_id": res["video_id"],
                    "video_path": res["video_path"],
                    "json_path": out_json,
                    "num_windows": len(res["windows"])
                })

                # liberar memoria entre vídeos
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] {vid} → {e}")

        # escribe un MANIFIESTO maestro que lista todos los JSON por vídeo
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"[OK] Guardado manifiesto: {manifest_path}")

