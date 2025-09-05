import os
import time
import torch
import random
import math
import shutil


from vlm_analyzer import DeepfakeVLMAnalyzer
from detection_pipeline import DeepfakeDetectionPipeline
from prompts import PROMPTS, INSTRUCTION
from VideoProcessor import VideoProcessor  # <-- si tu módulo es VideoProcessor.py, usa: from VideoProcessor import VideoProcessor
from dataset_builder import DatasetBuilder
from collections import defaultdict, Counter

# ============ CONFIGURACIÓN DE RUTAS ============
VIDEO_DIRS = [
    "/home/NAS/monicapina/Original_videos",
    "/home/NAS/monicapina/Manipulated_videos",
]

BASE_OUTPUT_DIR = "/home/mpina/repositories/old_truthlens/FFA_dataset"
FRAME_DIR = os.path.join(BASE_OUTPUT_DIR, "frames")
DATASET_JSON = os.path.join(BASE_OUTPUT_DIR, "dataset_cliplike.json")
DONE_FILE = os.path.join(BASE_OUTPUT_DIR, "processed_videos.txt")

# ============ CONFIG PIPELINE ============
NUM_FRAMES = 12
CONTEXT = 2# 👈 ventana temporal: 2*CONTEXT+1 (1 => trípletas; sube a 2 o 3 para 5 o 7 frames)
HOP = 2
CONFIDENCE_THRESHOLD = 0.7
DEEPFAKE_RATIO_THRESHOLD = 0.6
MODEL_NAME = "OpenGVLab/InternVL2_5-8B"
# Porcentaje de dataset a procesar (balanceado por clase/etnia/género)
SAMPLE_FRACTION = 0.30     # 30%
SAMPLE_SEED = 42           # reproducible
os.makedirs(FRAME_DIR, exist_ok=True)
random.seed(SAMPLE_SEED)
KEEP_FRAMES = True           # mientras no tengas el DataLoader, déjalo en True
KEEP_ONLY_ANALYZED = True  

# === helper: objetivos por atributo a partir de la etiqueta de vídeo ===
ATTR_KEYS = [
    "lighting_consistency","specular_highlights","skin_texture","eye_blinking",
    "eye_gaze_tracking","lip_motion_consistency","contour_blending","pose_3d",
    "temporal_artifacts","accessories_consistency","face_background_separation",
    "expressive_dynamics"
]

def make_attr_targets(expected_label: str):
    if expected_label == "deepfake":
        return {k: "suspicious" for k in ATTR_KEYS}
    if expected_label == "real":
        return {k: "realistic" for k in ATTR_KEYS}
    return None

# ============ FUNCIONES DE REANUDACIÓN ============
def load_done_videos(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f)

def mark_video_done(path, video_path: str):
    """
    Marca un vídeo como procesado guardando su ruta completa.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(video_path + "\n")

def expected_from_path(p: str) -> str:
    pl = p.lower()
    if "/original_videos/" in pl:
        return "real"
    if "/manipulated_videos/" in pl:
        return "deepfake"
    # fallback: usa VideoProcessor si quieres
    return "unknown"

def parse_ethnicity_gender(base_dir: str, full_path: str):
    """
    Asumiendo estructura:
      base_dir/
        Afro-descendant/Man/video.mp4
        (o similares)
    Devuelve (ethnicity, gender) o ("unknown","unknown") si no puede.
    """
    try:
        rel = os.path.relpath(full_path, base_dir)
        parts = rel.split(os.sep)
        # parts: [Ethnicity, Gender, file]
        ethnicity = parts[0] if len(parts) >= 1 else "unknown"
        gender = parts[1] if len(parts) >= 2 else "unknown"
        return ethnicity, gender
    except Exception:
        return "unknown", "unknown"


def collect_by_group(video_dirs):
    """
    Agrupa vídeos por (expected_label, ethnicity, gender).
    Devuelve: dict { (label, ethnicity, gender): [ (video_path, video_id) ] }
    """
    groups = defaultdict(list)
    for base in video_dirs:
        for root, _, files in os.walk(base):
            for fname in files:
                if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    continue
                video_path = os.path.join(root, fname)
                label = expected_from_path(video_path)
                ethnicity, gender = parse_ethnicity_gender(base, video_path)
                video_id = os.path.splitext(fname)[0]
                groups[(label, ethnicity, gender)].append((video_path, video_id))
    return groups

def balanced_sample(groups, frac: float, seed: int = 42):
    """
    Para cada grupo (clase, etnia, género) toma floor(len*frac) (>=1 si el grupo no está vacío).
    Mezcla con semilla fija para reproducibilidad.
    """
    rnd = random.Random(seed)
    selected = []
    summary = {}
    for key, items in groups.items():
        n = max(1, int(len(items) * frac))
        items_copy = items[:]
        rnd.shuffle(items_copy)
        pick = items_copy[:n]
        selected.extend(pick)
        summary[key] = {"total": len(items), "selected": len(pick)}
    return selected, summary

# ============ INICIALIZACIÓN ============
analyzer = DeepfakeVLMAnalyzer(
    MODEL_NAME,
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
builder = DatasetBuilder(DATASET_JSON)

done_videos = load_done_videos(DONE_FILE)



# ============ COLECCIÓN Y MUESTREO ============
groups = collect_by_group(VIDEO_DIRS)
selected, sel_summary = balanced_sample(groups, SAMPLE_FRACTION, SAMPLE_SEED)

print("\n📊 Balanceo (clase, etnia, género) -> total / seleccionados")
for (lbl, eth, gen), info in sorted(sel_summary.items()):
    print(f"  ({lbl}, {eth}, {gen}): {info['total']} -> {info['selected']}")
print(f"\n🧺 Vídeos seleccionados totales: {len(selected)}\n")
# ============ PROCESAMIENTO EN BUCLE ============
for base_dir in VIDEO_DIRS:
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            video_path = os.path.join(root, fname)
            video_id = os.path.splitext(fname)[0]

            if video_id in done_videos or builder.has_video(video_id):
                print(f"⏩ Skipping already processed video: {video_id}")
                continue

            print(f"\n🔍 Processing video: {video_id}")
            t_start = time.time()

            try:
                # Inicializa utilidades
                pipeline = DeepfakeDetectionPipeline(
                    video_path=video_path,
                    vlm_analyzer=analyzer,
                    prompts=PROMPTS,
                    instruction=INSTRUCTION,      
                    num_frames=NUM_FRAMES,
                    context=CONTEXT,  
                    hop=HOP,                   
                    conf_thresh=CONFIDENCE_THRESHOLD,
                    ratio_thresh=DEEPFAKE_RATIO_THRESHOLD
                )

                pipeline = DeepfakeDetectionPipeline(
                    video_path=video_path,
                    vlm_analyzer=analyzer,
                    prompts=PROMPTS,
                    instruction=INSTRUCTION,
                    num_frames=NUM_FRAMES,
                    context=CONTEXT,
                    hop=HOP,                         # <- nuevo
                    conf_thresh=CONFIDENCE_THRESHOLD,# <- ya lo tenías
                    ratio_thresh=DEEPFAKE_RATIO_THRESHOLD
                )

                video_processor = VideoProcessor(video_path, num_frames=NUM_FRAMES)
                expected = video_processor.get_expected_label()

                # 1) Guarda frames (una sola vez)
                output_frame_dir = os.path.join(FRAME_DIR, video_id)
                frame_paths = video_processor.save_frames_to_folder(output_frame_dir)

                # 2) Análisis por ventanas + prompts (instrucción guiada por GT dentro del pipeline)
                results = pipeline.run_analysis()

                # 3) Decisión global
                decision_info = pipeline.decide(results, expected_label=expected)
                decision = decision_info["predicted_label"]
                print(f"Decision: {decision}, Expected: {expected}")
                
                
                # 4) (Opcional) justificación narrativa si quieres
                summary = None
                if decision != "indeterminate":
                    summary = pipeline.generate_summary(results, decision)
                    if summary:
                        print("\n🧠 Justification:\n", summary.get("explanation", summary))
                
                # 5) Agregados por atributo y targets guiados
                attribute_summary = pipeline.summarize_attributes(results)
                attr_targets = pipeline.build_guided_attribute_targets(results, expected_label=expected)

                # 6) Guardado en el dataset
                '''
                builder.add_video(
                    video_id=video_id,
                    frame_paths=frame_paths,
                    frame_results=results,
                    expected_label=expected,
                    decision_info=decision_info,
                    video_caption=summary.get("explanation") if isinstance(summary, dict) else None,
                    attr_targets=attr_targets,
                    attribute_summary=attribute_summary,
                )
                '''
                per_attr = defaultdict(list)
                for r in results:
                    per_attr[r["center_frame"]].append(r)

                per_frame = {}
                for idx, rs in per_attr.items():
                    # voto por etiqueta del frame
                    labels = [ (rr.get("label") or "real") for rr in rs ]
                    pred = Counter(labels).most_common(1)[0][0]

                    # confianza media del frame (clamp por si faltan)
                    confs = [ float(rr.get("confidence") or 0.0) for rr in rs ]
                    conf_mean = round(sum(confs)/max(len(confs),1), 3)

                    # caption cortito con 1-2 evidencias más frecuentes
                    evid_pool = []
                    for rr in rs:
                        for e in (rr.get("evidence") or []):
                            if isinstance(e, str) and e.strip():
                                evid_pool.append(e.strip())
                    top_evid = [e for e, _ in Counter(evid_pool).most_common(2)]
                    if top_evid:
                        caption = "; ".join(top_evid)
                    else:
                        caption = f"Frame {idx} ({pred})"

                    per_frame[idx] = {
                        "caption": caption,
                        "confidence": conf_mean,
                        "predicted_label": pred
                    }
                builder.append_grouped_video(
                    video_id=video_id,
                    expected_label=expected,
                    decision_info=decision_info,
                    frame_paths=frame_paths,
                    per_frame=per_frame,                 # <- sólo los índices analizados
                    video_caption=summary.get("explanation") if summary else None,
                    attribute_summary=attribute_summary, # opcional
                    attr_targets=attr_targets,            # opcional
                    extra_meta={
                        "source_video_path": video_path,
                        "sampling_params": {"num_frames": NUM_FRAMES, "context": CONTEXT, "hop": HOP, "strategy": "uniform"},
                        "frames_stored": KEEP_FRAMES
                    }
                )


                builder.save()
                mark_video_done(DONE_FILE, video_path)

                if not KEEP_FRAMES:
                    try:
                        shutil.rmtree(output_frame_dir, ignore_errors=True)
                        print(f"🧹 Deleted frames folder: {output_frame_dir}")
                    except Exception as e:
                        print(f"⚠️ Could not delete {output_frame_dir}: {e}")
                elif KEEP_ONLY_ANALYZED:
                    # Borrar frames no analizados (deja solo los índices en per_frame)
                    keep_set = set(per_frame.keys())
                    try:
                        for fn in os.listdir(output_frame_dir):
                            # asumiendo nombres frame_0007.jpg → parsear índice:
                            if fn.startswith("frame_") and fn.endswith(".jpg"):
                                idx = int(fn[6:10])  # ajusta si tu padding es otro
                                if idx not in keep_set and os.path.isfile(os.path.join(output_frame_dir, fn)):
                                    os.remove(os.path.join(output_frame_dir, fn))
                        print(f"🧹 Kept only analyzed frames in: {output_frame_dir}")
                    except Exception as e:
                        print(f"⚠️ Could not prune frames in {output_frame_dir}: {e}")

            except Exception as e:
                print(f"❌ Error processing {video_id}: {e}")

    print(f"\n📦 Full dataset saved to {DATASET_JSON}")
