import os
import json
from tqdm import tqdm
from VideoProcessor import VideoProcessor
from collections import Counter


class DeepfakeDetectionPipeline:
    def __init__(self, video_path, vlm_analyzer, prompts, instruction, num_frames=5, conf_thresh=0.7, ratio_thresh=0.6):
        self.video_path = video_path
        self.vlm_analyzer = vlm_analyzer
        self.prompts = prompts
        self.instruction = instruction
        self.num_frames = num_frames
        self.conf_thresh = conf_thresh
        self.ratio_thresh = ratio_thresh

    def run_analysis(self):
        video_processor = VideoProcessor(self.video_path, num_frames=self.num_frames)
        frames = video_processor.extract_frames()
        results = []

        #for i in tqdm(range(len(frames))):
        for i in tqdm(range(1, len(frames) - 1)):

            frame_triplet = []
            if i > 0: frame_triplet.append(frames[i - 1])
            frame_triplet.append(frames[i])
            if i < len(frames) - 1: frame_triplet.append(frames[i + 1])

            for prompt in self.prompts:
                question = self.instruction + "\n" + prompt
                response = self.vlm_analyzer.query(frame_triplet, question)
                print(f"\nFrame {i}: {prompt}\n{json.dumps(response, indent=2)}")
                results.append({
                    "frame": i,
                    "prompt": prompt,
                    "label": response.get("label", ""),
                    "confidence": response.get("confidence"),
                    "explanation": response.get("explanation", "")
                })

        return results

    def save_results(self, results, filename):
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", filename), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to 'output/{filename}'")

    def decide(self, results, expected_label: str = None):
        deep = [r for r in results if r["label"] == "deepfake"]
        real = [r for r in results if r["label"] == "real"]
        total = len(deep) + len(real)

        if total == 0:
            return {
                "predicted_label": "indeterminate",
                "confidence_score": 0.0,
                "expected_label": expected_label
            }

        ratio = len(deep) / total
        predicted = "deepfake" if ratio >= self.ratio_thresh else "real"
        score = ratio if predicted == "deepfake" else 1.0 - ratio
        
        return {
            "predicted_label": predicted,
            "confidence_score": round(score, 3),
            "expected_label": expected_label
    }

    

    from collections import Counter

    def generate_summary(self, results: list[dict], decision: str) -> dict | None:
        explanations = [
            r["explanation"] for r in results
            if r["label"] == decision and isinstance(r["confidence"], float) and r["confidence"] >= self.conf_thresh
        ]
        if not explanations:
            return None

        # Paso 1: Agrupar frases frecuentes y únicas
        normalized_clues = [e.lower().strip().rstrip(".") for e in explanations if len(e) > 20]
        clue_counts = Counter(normalized_clues)
        top_clues = [f"{text}." for text, count in clue_counts.items() if count >= 1]

        # Paso 2: Frame de contexto visual
        from VideoProcessor import VideoProcessor
        video_processor = VideoProcessor(self.video_path, num_frames=self.num_frames)
        frames = video_processor.extract_frames()
        mid_frame = frames[len(frames) // 2]

        # Paso 3: Generar frases técnicas
        bullet_prompt = (
            f"You are a deepfake forensic analyst working with visual-text contrastive models like CLIP.\n"
            f"Your task is to rewrite the following forensic clues into multiple short, technical sentences.\n"
            f"Use forensic vocabulary (e.g., skin texture, lighting gradients, facial landmarks).\n"
            f"Include spatial references (e.g., forehead, jaw, eyes).\n"
            f"Add negations where appropriate (e.g., 'no signs of...', 'not visible...').\n"
            f"If possible, rephrase clues using equivalent but varied wording to reinforce key ideas.\n"
            f"Output should be a list of short, independent sentences.\n\n"
            f"Clues:\n" + "\n".join(f"- {c}" for c in top_clues)
        )

        bullet_response = self.vlm_analyzer.query([mid_frame], bullet_prompt)
        bullet_text = bullet_response.get("explanation", bullet_response)

        # Paso 4: Síntesis narrativa
        summary_prompt = (
            "You are a professional forensic writing assistant.\n"
            "Your task is to summarize the following technical observations into one single paragraph.\n"
            "Avoid repeating the same sentence structure or phrases.\n"
            "Use varied vocabulary and merge similar clues into cohesive statements.\n"
            "Include spatial details (e.g., forehead, jaw, eyes) and forensic language (e.g., pores, gradients, artifacts).\n"
            "Only write one paragraph that sounds natural, fluent, and technically accurate.\n"
            f"The paragraph should explain why the video is likely {decision.upper()}.\n\n"
            "Observations:\n" + bullet_text
        )


        final_response = self.vlm_analyzer.query([mid_frame], summary_prompt)
        return final_response



