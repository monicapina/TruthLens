# dataset_builder.py
import os
import json
import time

class DatasetBuilder:
    """
    Mantiene un JSON con estructura por VÍDEO:
    {
      "dataset_name": "TruthLens-VLM",
      "created_at": "...",
      "videos": [
        {
          "video_id": "...",
          "expected_label": "...",
          "global_prediction": "...",
          "global_confidence": ...,
          "video_caption": "...",
          "frames": [
            {
              "frame_index": 0,
              "image_path": "...",
              "triplet_paths": ["...", "...", "..."],
              "caption": "...",
              "confidence": 0.0,
              "predicted_label": "...",
              "expected_label": "..."
            }
          ],
          "meta": {
            "attribute_summary": {...},
            "attr_targets": {...}
          }
        }
      ]
    }
    """

    def __init__(self, output_json_path: str):
        self.output_path = output_json_path
        self.data = self._load_or_init(output_json_path)
        self._video_index = {v.get("video_id") for v in self.data.get("videos", []) if v.get("video_id")}

    def _load_or_init(self, path: str) -> dict:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                # Si era un JSON plano (lista legacy), lo envolvemos:
                if isinstance(obj, list):
                    obj = {
                        "dataset_name": "TruthLens-VLM",
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "videos": obj
                    }
                if not isinstance(obj, dict):
                    obj = {}
                obj.setdefault("dataset_name", "TruthLens-VLM")
                obj.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
                obj.setdefault("videos", [])
                return obj
            except Exception:
                pass

        # Si no existe o falló la lectura, creamos uno nuevo
        return {
            "dataset_name": "TruthLens-VLM",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos": []
        }

    # ----------------- utilidades públicas -----------------

    def has_video(self, video_id: str) -> bool:
        return video_id in self._video_index

    def _get_triplet(self, frames: list[str], index: int) -> list[str]:
        triplet = []
        if index > 0:
            triplet.append(frames[index - 1])
        triplet.append(frames[index])
        if index < len(frames) - 1:
            triplet.append(frames[index + 1])
        return triplet

    def append_grouped_video(
        self,
        *,
        video_id: str,
        expected_label: str | None,
        decision_info: dict | None,
        frame_paths: list[str],
        per_frame: dict[int, dict] | None = None,
        video_caption: str | None = None,
        attribute_summary: dict | None = None,
        attr_targets: dict | None = None,
        extra_meta: dict | None = None,
    ) -> None:
        """
        Crea (o reemplaza) una entrada por vídeo con sus frames anidados.

        🚩 Cambio clave: si 'per_frame' viene informado, SOLO se guardan en el JSON
        los índices presentes en 'per_frame' (i.e., los frames realmente evaluados).
        Si 'per_frame' es None o vacío, se hace fallback y se guardan todos los frames.

        per_frame: {idx: {"caption":..., "confidence":..., "predicted_label":..., "triplet_paths":[...]}}

        decision_info: {"predicted_label":..., "confidence_score":...}
        """
        # Info global
        global_pred = None
        global_conf = None
        if isinstance(decision_info, dict):
            global_pred = decision_info.get("predicted_label")
            global_conf = decision_info.get("confidence_score")

        # ¿Qué índices guardamos?
        N = len(frame_paths)
        if per_frame:
            indices = sorted(int(k) for k in per_frame.keys() if 0 <= int(k) < N)
        else:
            indices = list(range(N))  # fallback clásico

        # Construir frames SOLO para los índices analizados
        frames = []
        for idx in indices:
            img_path = frame_paths[idx]
            pf = (per_frame or {}).get(idx, {})

            # tripleta: si no te pasan una ya hecha, la construimos con vecinos
            triplet = pf.get("triplet_paths")
            if not triplet:
                triplet = []
                if idx - 1 >= 0:
                    triplet.append(frame_paths[idx - 1])
                triplet.append(img_path)
                if idx + 1 < N:
                    triplet.append(frame_paths[idx + 1])

            caption = pf.get("caption") or f"Frame {idx} processed (see per-attribute evidence)."
            confidence = float(pf.get("confidence")) if pf.get("confidence") is not None else 0.0
            pred = pf.get("predicted_label") or global_pred or "indeterminate"

            frames.append({
                "frame_index": idx,
                "image_path": img_path,
                "triplet_paths": triplet,
                "caption": caption,
                "confidence": confidence,
                "predicted_label": pred,
                "expected_label": expected_label,
            })

        # Entrada del vídeo
        entry = {
            "video_id": video_id,
            "expected_label": expected_label,
            "global_prediction": global_pred,
            "global_confidence": float(global_conf) if global_conf is not None else None,
            "video_caption": video_caption,
            "frames": frames,
            "meta": {}
        }
        if attribute_summary is not None:
            entry["meta"]["attribute_summary"] = attribute_summary
        if attr_targets is not None:
            entry["meta"]["attr_targets"] = attr_targets
        if extra_meta:
            entry["meta"].update(extra_meta)

        # Reemplazar si existía y guardar índice
        self.data["videos"] = [v for v in self.data["videos"] if v.get("video_id") != video_id]
        self.data["videos"].append(entry)
        self._video_index.add(video_id)


    def add_video(
        self,
        video_id: str,
        frame_paths: list[str],
        frame_results: list[dict],
        expected_label: str,
        decision_info: dict,
        video_caption: str | None = None,
        attr_targets: dict | None = None,
        attribute_summary: dict | None = None,
    ) -> None:
        """
        Versión 'legacy' que antes generaba entradas planas por frame.
        Ahora crea también una ENTRADA AGRUPADA por vídeo.
        - Si frame_results tiene 1-a-1 con frame_paths, usamos esos campos.
        - Si frame_results es más largo (p.ej. por atributos/prompts), hacemos
          un fallback con caption=..., confidence=..., predicted_label=...
          si existen en cada item; si no, usamos la global.
        """
        # Info global
        global_pred = decision_info.get("predicted_label") if isinstance(decision_info, dict) else None
        global_conf = decision_info.get("confidence_score") if isinstance(decision_info, dict) else None

        frames = []
        N = len(frame_paths)

        # Intento de mapeo 1-a-1 primero
        one_to_one = len(frame_results) == N

        for idx, img_path in enumerate(frame_paths):
            triplet = self._get_triplet(frame_paths, idx)

            if one_to_one:
                fr = frame_results[idx] or {}
            else:
                # Si no es 1-a-1, intentamos encontrar algún resultado del mismo frame
                # (por ejemplo, el primero que tenga "center_frame" == idx) o usamos {}
                fr = next((r for r in frame_results if r.get("frame") == idx or r.get("center_frame") == idx), {})

            caption = fr.get("caption") or fr.get("explanation")
            if not caption:
                # si el resultado viene en forma de evidencia lista
                ev = fr.get("evidence")
                if isinstance(ev, list) and ev:
                    caption = "; ".join(map(str, ev))
            if not caption:
                caption = f"Frame {idx} processed."

            conf = fr.get("confidence")
            confidence = float(conf) if isinstance(conf, (int, float, str)) and str(conf) not in ("", "None") else 0.0

            pred = fr.get("label") or fr.get("predicted_label") or global_pred or "indeterminate"

            frames.append({
                "frame_index": idx,
                "image_path": img_path,
                "triplet_paths": triplet,
                "caption": caption,
                "confidence": confidence,
                "predicted_label": pred,
                "expected_label": expected_label,
            })

        entry = {
            "video_id": video_id,
            "expected_label": expected_label,
            "global_prediction": global_pred,
            "global_confidence": float(global_conf) if global_conf is not None else None,
            "video_caption": video_caption,
            "frames": frames,
            "meta": {}
        }
        if attribute_summary is not None:
            entry["meta"]["attribute_summary"] = attribute_summary
        if attr_targets is not None:
            entry["meta"]["attr_targets"] = attr_targets

        # Reemplazar si existía
        self.data["videos"] = [v for v in self.data["videos"] if v.get("video_id") != video_id]
        self.data["videos"].append(entry)
        self._video_index.add(video_id)

    def save(self) -> None:
        out_dir = os.path.dirname(self.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
