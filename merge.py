import json
import argparse
from datetime import datetime
'''
python merge.py --json1 /home/mpina/repositories/old_truthlens/FFA_dataset/dataset_cliplike.json --json2 /home/mpina/repositories/old_truthlens/FFA_dataset/dataset_cliplike_2.json --out /home/mpina/repositories/old_truthlens/FFA_dataset/merged.json
'''
def merge_jsons(json1_path, json2_path, out_path, dataset_name="TruthLens-VLM-Merged"):
    # Cargar ambos
    with open(json1_path, 'r') as f1, open(json2_path, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Estructura de salida
    merged = {
        "dataset_name": dataset_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "videos": []
    }

    # Añadir videos
    merged["videos"].extend(data1.get("videos", []))
    merged["videos"].extend(data2.get("videos", []))

    # Guardar
    with open(out_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"Combinados {len(data1.get('videos', []))} + {len(data2.get('videos', []))} vídeos → {len(merged['videos'])} en {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json1", type=str, required=True, help="Primer JSON")
    ap.add_argument("--json2", type=str, required=True, help="Segundo JSON")
    ap.add_argument("--out", type=str, required=True, help="Archivo de salida")
    args = ap.parse_args()

    merge_jsons(args.json1, args.json2, args.out)

