import os
import json
from tqdm import tqdm
from VideoProcessor import VideoProcessor
from collections import Counter, defaultdict
import math
# --- Mapeo por substring del prompt -> clave de atributo
PROMPT2ATTR = {
    "lighting/shadow": "lighting_consistency",
    "skin texture": "skin_texture",
    "eye-related": "eye_blinking",              # si separas gaze, añade otro mapeo
    "mouth/lip/teeth": "lip_motion_consistency",
    "contour/blending": "contour_blending",
    "head pose/3d": "pose_3d",
    "temporal artifacts": "temporal_artifacts",
    "face-background": "face_background_separation",
    "accessories consistency": "accessories_consistency",
    "expressive dynamics": "expressive_dynamics",
}

def _attr_from_prompt(prompt: str) -> str | None:
    p = (prompt or "").lower()
    for key, attr in PROMPT2ATTR.items():
        if key in p:
            return attr
    return None

def make_guided_instruction(gt_label: str) -> str:
    gt_label = (gt_label or "").lower().strip()
    if gt_label == "deepfake":
        hint = (
            "Ground-truth supervision: DEEPFAKE. Be EXTRA vigilant. "
            "If you notice even subtle forensic cues of manipulation, set verdict='suspicious'. "
            "If you do NOT find any concrete cue for this attribute, set verdict='realistic'."
        )
    elif gt_label == "real":
        hint = (
            "Ground-truth supervision: REAL. Be vigilant for mismatches, "
            "but if you do NOT find concrete issues for this attribute, set verdict='realistic'. "
            "Only set 'suspicious' with visible, specific cues."
        )
    else:
        hint = (
            "Ground-truth is unknown. Make a neutral decision strictly from the visible evidence."
        )

    # Rúbrica de confianza para evitar valores fijos
    rubric = (
        "Confidence rubric:\n"
        "- 0.20–0.39: weak or unclear evidence.\n"
        "- 0.40–0.59: mixed/partial evidence.\n"
        "- 0.60–0.79: clear evidence with minor uncertainty.\n"
        "- 0.80–0.95: strong, repeated, multi-frame evidence.\n"
        "Never output 1.0.\n"
    )

    # Devuelve SOLO un objeto JSON simple
    return (
        "You are a forensic analyst for deepfake detection.\n"
        f"{hint}\n"
        "Given a short window of consecutive frames, analyze ONLY the requested attribute.\n"
        f"{rubric}"
        "Return STRICTLY this JSON object (no explanations outside JSON):\n"
        "{\n"
        '  \"attribute\": \"<attr_key>\",\n'
        '  \"verdict\": \"suspicious\" | \"realistic\",\n'
        '  \"confidence\": 0.0–1.0,\n'
        '  \"evidence\": [\"short bullet 1\", \"short bullet 2\"]\n'
        "}\n"
        "Keep bullets short, concrete, and visual/forensic (lighting, texture, blink, blending, temporal artifacts)."
    )

# --- pesos por atributo (ajustables)
DEFAULT_ATTR_WEIGHTS = {
    "lighting_consistency": 1.4,
    "lip_motion_consistency": 1.4,
    "expressive_dynamics": 1.3,
    "contour_blending": 1.2,
    "pose_3d": 1.1,
    # el resto 1.0 por defecto
}

class DeepfakeDetectionPipeline:
    def __init__(
        self,
        video_path,
        vlm_analyzer,
        prompts,
        instruction,
        num_frames=48,
        context=1,
        hop=1,
        conf_thresh=0.75,
        ratio_thresh=0.5
    ):
        self.video_path = video_path
        self.vlm_analyzer = vlm_analyzer
        self.prompts = prompts
        self.instruction = instruction
        self.num_frames = num_frames
        self.context = max(0, int(context))
        self.hop = max(1, int(hop))
        self.conf_thresh = conf_thresh
        self.ratio_thresh = ratio_thresh


    def _make_window(self, frames, center_idx):
        left = max(0, center_idx - self.context)
        right = min(len(frames) - 1, center_idx + self.context)
        return frames[left:right+1]

    def run_analysis(self):
        video_processor = VideoProcessor(self.video_path, num_frames=self.num_frames)
        frames = video_processor.extract_frames()
        if not frames:
            return []

        # etiqueta GT del vídeo para guiar la instrucción
        expected_label = video_processor.get_expected_label()  # "real" | "deepfake" | "unknown"
        guided_instruction = make_guided_instruction(expected_label)

        results = []
        start = self.context
        end = len(frames) - self.context - 1
        if end < start:
            start, end = 0, len(frames) - 1

        for i in range(start, end + 1, self.hop):
            # ventana temporal alrededor del frame central
            #left = max(0, i - self.context)
            #right = min(len(frames) - 1, i + self.context)
            #window_frames = frames[left:right+1]
            window_frames = self._make_window(frames, i)

            for prompt in self.prompts:
                attr_key = _attr_from_prompt(prompt) or "unknown_attribute"
                question = (
                    f"{guided_instruction}\n"
                    f'Attribute key: "{attr_key}"\n'
                    f"Task: {prompt}"
                )

                response = self.vlm_analyzer.query(window_frames, question)

                # --- normalización robusta (sin contradicciones)
                if isinstance(response, dict):
                    verdict = (response.get("verdict") or "realistic").lower()
                    try:
                        conf = float(response.get("confidence", 0.0))
                    except Exception:
                        conf = 0.0
                    if math.isnan(conf):
                        conf = 0.0
                    evid = response.get("evidence") or []
                    # Forzar atributo estable del prompt
                    if not response.get("attribute"):
                        response["attribute"] = attr_key
                else:
                    verdict, conf, evid = "realistic", 0.0, [str(response)]

                # clamp + pequeña post-calibración para evitar 0.85 fijo
                if not isinstance(evid, list):
                    evid = [str(evid)]
                num_ev = len([e for e in evid if isinstance(e, str) and e.strip()])

                conf = max(0.2, min(float(conf), 0.95))
                if verdict == "suspicious":
                    if num_ev >= 2:
                        conf = min(0.95, conf + 0.06)
                    elif num_ev == 0:
                        conf = min(conf, 0.45)
                else:
                    if num_ev == 0:
                        conf = min(conf, 0.6)

                label_bin = "deepfake" if verdict == "suspicious" else "real"

                results.append({
                    "center_frame": i,
                    "window_size": len(window_frames),
                    "prompt": prompt,
                    "attribute": attr_key,
                    "verdict": verdict,         # suspicious | realistic
                    "label": label_bin,         # deepfake | real (para decide())
                    "confidence": conf,
                    "evidence": evid,           # SOLO evidencias (lista corta)
                })

                if label_bin == expected_label:

                    print(f"Frame {i}, Attr '{attr_key}': Verdict={verdict}, Conf={conf:.2f}")
                    print("result:", results[-1])  # Depuración detallada

        return results

    def save_results(self, results, filename):
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", filename), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to 'output/{filename}'")

    def decide(self, results, expected_label: str = None):
        grouped = defaultdict(lambda: {"deepfake": [], "real": []})
        for r in results:
            lab = (r.get("label") or "").lower()
            try:
                conf = float(r.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            # filtra ruido por confianza mínima si quieres mantenerlo
            if conf < self.conf_thresh:
                continue
            attr = r.get("attribute") or "unknown_attribute"
            grouped[attr][lab].append(conf)

        deep_w = 0.0
        real_w = 0.0

        # ⚠️ Sin Top-K: sumamos TODAS las confianzas (puedes dejar los pesos o poner todos a 1.0)
        DEFAULT_ATTR_WEIGHTS = {
            "lighting_consistency": 1.4,
            "lip_motion_consistency": 1.4,
            "expressive_dynamics": 1.3,
            "contour_blending": 1.2,
            "pose_3d": 1.1,
        }

        for attr, buckets in grouped.items():
            w_attr = DEFAULT_ATTR_WEIGHTS.get(attr, 1.0)  # pon 1.0 si no quieres pesos
            if buckets["deepfake"]:
                deep_w += sum(c * w_attr for c in buckets["deepfake"])
            if buckets["real"]:
                real_w += sum(c * w_attr for c in buckets["real"])

        total = deep_w + real_w
        if total == 0.0:
            # fallback: mayoría simple si todo quedó filtrado
            deep = sum(1 for r in results if (r.get("label") or "").lower() == "deepfake")
            real = sum(1 for r in results if (r.get("label") or "").lower() == "real")
            total_cnt = deep + real
            if total_cnt == 0:
                return {"predicted_label": "indeterminate", "confidence_score": 0.0, "expected_label": expected_label}
            ratio = deep / total_cnt
        else:
            ratio = deep_w / total

        predicted = "deepfake" if ratio >= self.ratio_thresh else "real"
        score = ratio if predicted == "deepfake" else 1.0 - ratio
        return {
            "predicted_label": predicted,
            "confidence_score": round(float(score), 3),
            "expected_label": expected_label
        }



    def summarize_attributes(self, results: list[dict]) -> dict:
        acc = defaultdict(lambda: {"susp": 0.0, "real": 0.0})
        for r in results:
            attr = r.get("attribute") or "unknown_attribute"
            #conf = float(r.get("confidence") or 0.0)
            try:
                conf = float(r.get("confidence") or 0.0)
            except Exception:
                conf = 0.0

            verdict = (r.get("verdict") or "undetermined").lower()
            if verdict == "suspicious":
                acc[attr]["susp"] += conf
            elif verdict == "realistic":
                acc[attr]["real"] += conf

        out = {}
        for attr, d in acc.items():
            tot = d["susp"] + d["real"]
            p_susp = d["susp"] / tot if tot > 0 else 0.0
            out[attr] = {
                "weighted_suspicious_score": round(p_susp, 4),
                "label": "suspicious" if p_susp >= 0.5 else "realistic",
                "total_weight": round(tot, 4)
            }
        return out

    def build_guided_attribute_targets(self, results: list[dict], expected_label: str) -> dict:
        """
        Repondera los pesos hacia la clase real del vídeo (suave).
        Devuelve {attr: p_susp} en [0..1].
        """
        acc = defaultdict(lambda: {"susp": 0.0, "real": 0.0})
        for r in results:
            attr = r.get("attribute") or "unknown_attribute"
            #conf = float(r.get("confidence") or 0.0)
            try:
                conf = float(r.get("confidence") or 0.0)
            except Exception:
                conf = 0.0

            verdict = (r.get("verdict") or "undetermined").lower()
            if verdict == "suspicious":
                acc[attr]["susp"] += conf
            elif verdict == "realistic":
                acc[attr]["real"] += conf

        # guiado leve por GT
        alpha_pos, alpha_neg = 1.25, 0.75
        guided = {}
        for attr, d in acc.items():
            s, g = d["susp"], d["real"]
            if expected_label == "deepfake":
                s *= alpha_pos; g *= alpha_neg
            elif expected_label == "real":
                g *= alpha_pos; s *= alpha_neg
            tot = s + g
            guided[attr] = float(s / tot) if tot > 0 else 0.0
        return guided

    def generate_summary(self, results: list[dict], decision: str) -> dict | None:
        # Nota: tus items actuales no llevan "observations";
        # mantenemos este método por compatibilidad, pero puede devolver None.
        all_obs = []
        for r in results:
            for o in r.get("observations", []):
                if isinstance(o, str) and len(o) > 3:
                    all_obs.append(o.lower().strip().rstrip("."))

        if not all_obs:
            return None

        clue_counts = Counter(all_obs)
        top_clues = [f"{c}." for c, _ in clue_counts.items()]

        video_processor = VideoProcessor(self.video_path, num_frames=self.num_frames)
        frames = video_processor.extract_frames()
        if not frames:
            return None
        mid_frame = frames[len(frames)//2]

        bullet_prompt = (
            "You are a deepfake forensic analyst working with visual-text contrastive models like CLIP.\n"
            "Rewrite the following forensic clues into SHORT, technical bullet points.\n"
            "Use forensic vocabulary (skin texture, lighting gradients, facial landmarks), spatial references (forehead, jaw, eyes),\n"
            "and include negations where appropriate ('no signs of...', 'not visible...').\n"
            "Output ONLY a bullet list.\n\n"
            "Clues:\n" + "\n".join(f"- {c}" for c in top_clues)
        )
        bullet_response = self.vlm_analyzer.query([mid_frame], bullet_prompt)
        bullet_text = bullet_response.get("observations") or bullet_response.get("explanation") or str(bullet_response)

        summary_prompt = (
            "You are a professional forensic writing assistant.\n"
            "Summarize the following technical observations in ONE paragraph, natural and precise.\n"
            "Use varied wording, merge similar clues, and keep spatial/forensic terms.\n"
            "Do NOT classify real/deepfake; only summarize observations.\n\n"
            "Observations:\n" + ( "\n".join(bullet_text) if isinstance(bullet_text, list) else str(bullet_text) )
        )
        final_response = self.vlm_analyzer.query([mid_frame], summary_prompt)
        return final_response
