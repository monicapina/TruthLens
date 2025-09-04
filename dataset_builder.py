import os
import json

class DatasetBuilder:
    def __init__(self, output_json_path):
        self.output_path = output_json_path
        self.entries = []
        self._load_existing()  # para continuar si ya existe

    def _load_existing(self):
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                self.entries = json.load(f)

    def add_video(self, video_id, frame_paths, frame_results, expected_label, decision_info, video_caption=None):
        for i, (frame_path, frame_result) in enumerate(zip(frame_paths, frame_results)):
            triplet_paths = self._get_triplet(frame_paths, i)

            self.entries.append({
                "image_path": frame_path,
                "triplet_paths": triplet_paths,  # 👈 nuevo campo
                "caption": frame_result["explanation"],
                "confidence": frame_result["confidence"],
                "predicted_label": frame_result["label"],
                "expected_label": expected_label,
                "video_id": video_id,
                "frame_index": i,
                "video_caption": video_caption,
                "global_prediction": decision_info["predicted_label"],
                "global_confidence": decision_info["confidence_score"]
            })


    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    def has_video(self, video_id):
        return any(entry["video_id"] == video_id for entry in self.entries)

    def _get_triplet(self, frames, index):
        triplet = []
        if index > 0:
            triplet.append(frames[index - 1])
        triplet.append(frames[index])
        if index < len(frames) - 1:
            triplet.append(frames[index + 1])
        return triplet
    