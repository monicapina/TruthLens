import os
import time
import torch

from vlm_analyzer import DeepfakeVLMAnalyzer
from detection_pipeline import DeepfakeDetectionPipeline
from prompts import PROMPTS, INSTRUCTION
from VideoProcessor import VideoProcessor
from dataset_builder import DatasetBuilder

# ============ CONFIGURACIÓN DE RUTAS ============
VIDEO_DIR = "/home/mpina/repositories/old_truthlens/Data"
BASE_OUTPUT_DIR = "/home/mpina/repositories/old_truthlens/Dataset"
FRAME_DIR = os.path.join(BASE_OUTPUT_DIR, "frames")
DATASET_JSON = os.path.join(BASE_OUTPUT_DIR, "dataset_cliplike.json")
DONE_FILE = os.path.join(BASE_OUTPUT_DIR, "processed_videos.txt")

NUM_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.7
DEEPFAKE_RATIO_THRESHOLD = 0.6
MODEL_NAME = "OpenGVLab/InternVL2_5-8B"

os.makedirs(FRAME_DIR, exist_ok=True)

# ============ FUNCIONES DE REANUDACIÓN ============
def load_done_videos(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f)

def mark_video_done(path, video_id):
    with open(path, "a") as f:
        f.write(video_id + "\n")

# ============ INICIALIZACIÓN ============
analyzer = DeepfakeVLMAnalyzer(
    MODEL_NAME,
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
builder = DatasetBuilder(DATASET_JSON)
done_videos = load_done_videos(DONE_FILE)

# ============ PROCESAMIENTO EN BUCLE ============
for root, _, files in os.walk(VIDEO_DIR):
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
            pipeline = DeepfakeDetectionPipeline(
                video_path=video_path,
                vlm_analyzer=analyzer,
                prompts=PROMPTS,
                instruction=INSTRUCTION,
                num_frames=NUM_FRAMES,
                conf_thresh=CONFIDENCE_THRESHOLD,
                ratio_thresh=DEEPFAKE_RATIO_THRESHOLD
            )

            video_processor = VideoProcessor(video_path, num_frames=NUM_FRAMES)
            results = pipeline.run_analysis()

            expected = video_processor.get_expected_label()
            decision_info = pipeline.decide(results, expected_label=expected)
            decision = decision_info["predicted_label"]

            summary = None
            if decision != "indeterminate":
                summary = pipeline.generate_summary(results, decision)
                if summary:
                    print("\n🧠 Justification:\n", summary.get("explanation", summary))

            # Paso 1: Guardar los frames a disco primero
            output_frame_dir = os.path.join(FRAME_DIR, video_id)
            frame_paths = video_processor.save_frames_to_folder(output_frame_dir)

            # Paso 2: Ejecutar el análisis sobre los frames ya generados
            results = pipeline.run_analysis()

            builder.add_video(
                video_id=video_id,
                frame_paths=frame_paths,
                frame_results=results,
                expected_label=expected,
                decision_info=decision_info,
                video_caption=summary.get("explanation") if summary else None
            )
            builder.save()
            mark_video_done(DONE_FILE, video_id)

            print(f"✅ Video {video_id} processed in {time.time() - t_start:.2f}s")

        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")

print(f"\n📦 Full dataset saved to {DATASET_JSON}")
