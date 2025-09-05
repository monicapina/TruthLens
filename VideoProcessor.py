import numpy as np
import av
from PIL import Image
import os
class VideoProcessor:
    def __init__(self, video_path, frame_size=(448, 448), num_frames=5):
        self.video_path = video_path
        self.frame_size = frame_size
        self.num_frames = num_frames

    def extract_frames(self):
        container = av.open(self.video_path)
        total = container.streams.video[0].frames
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]: break
            if i in indices:
                frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")).resize(self.frame_size))
        #print(f"✅ Extracted {len(frames)} frames.")
        return frames
    
    def save_frames_to_folder(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        frames = self.extract_frames()
        saved_paths = []

        for idx, frame in enumerate(frames):
            frame_filename = f"frame_{idx:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            frame.save(frame_path)
            saved_paths.append(frame_path)

        #print(f"🖼️ Saved {len(saved_paths)} frames to '{output_dir}'")
        return saved_paths

    def get_expected_label(self):
        """
        Extrae la etiqueta esperada ('real' o 'deepfake') a partir de la ruta del vídeo.
        - 'Original_videos'  -> real
        - 'Manipulated_videos' -> deepfake
        """
        path_lower = self.video_path.lower()

        if "original_videos" in path_lower:
            return "real"
        elif "manipulated_videos" in path_lower:
            return "deepfake"
        else:
            return "unknown"
        
    def get_ethnicity_gender(self):
        """
        Extrae (ethnicity, gender) de la ruta:
        .../(Original_videos|Manipulated_videos)/<Ethnicity>/<Gender>/file.mp4
        """
        parts = self.video_path.replace("\\", "/").split("/")
        try:
            # Busca el índice del folder raíz
            if "Original_videos" in parts:
                i = parts.index("Original_videos")
            else:
                i = parts.index("Manipulated_videos")
            ethnicity = parts[i + 1] if i + 1 < len(parts) else None
            gender = parts[i + 2] if i + 2 < len(parts) else None
        except ValueError:
            ethnicity, gender = None, None
        return ethnicity, gender

