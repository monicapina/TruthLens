import json
import argparse

def build_pairs(input_json, output_jsonl):
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    videos = dataset.get("videos", [])
    total_pairs = 0

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for video in videos:
            video_id = video.get("video_id")
            video_path = video.get("video_path")
            video_label = video.get("summary", {}).get("heuristic_decision", {}).get("label", None)

            for window in video.get("windows", []):
                window_id = window.get("window_id")
                t_start = window.get("t_start")
                t_end = window.get("t_end")

                for attr_key, attr_data in window.get("attributes", {}).items():
                    text = f"[{attr_key}] {attr_data.get('explanation', '')}"
                    label = video_label  # usa la decisión heurística global del vídeo
                    confidence = attr_data.get("confidence_mean", None)

                    entry = {
                        "video_id": video_id,
                        "video_path": video_path,
                        "window_id": window_id,
                        "t_start": t_start,
                        "t_end": t_end,
                        "attribute": attr_key,
                        "text": text.strip(),
                        "label": label,
                        "confidence": confidence
                    }

                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_pairs += 1

    print(f"[OK] Guardado {output_jsonl} con {total_pairs} ejemplos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construir dataset JSONL a partir de dataset.json")
    parser.add_argument("--input", required=True, help="Ruta al dataset.json")
    parser.add_argument("--output", default="pairs.jsonl", help="Ruta al JSONL de salida")
    args = parser.parse_args()

    build_pairs(args.input, args.output)
