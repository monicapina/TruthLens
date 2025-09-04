'''
PROMPTS = [
    "Are there inconsistencies or abrupt changes in lighting direction or intensity on the face between frames, especially near the hairline, jaw, or nose?",
    "Do shadows on the face and neck shift abruptly or seem detached from the underlying facial structure?",
    "Does the lighting on the face feel mismatched compared to the background or environmental light sources?",
    "Are there areas on the skin where highlights appear unnaturally static, uniform, or unnaturally sharp between frames?",
    "Does the skin texture remain consistent across time, especially on the cheeks, forehead, and chin?",
    "Do some frames show skin that appears overly smooth, glossy, plastic-like, or has lost detail in pores or wrinkles?",
    "Do pores, blemishes, moles, or wrinkles appear or disappear abruptly or shift position between frames?",
    "Are there unnatural transitions or flickering in facial texture details such as cheeks, forehead, or around the eyes?",
    "Do the eyes blink at realistic and variable intervals, or do they blink too frequently, too rarely, or in a robotic manner?",
    "Are the eye movements fluid and responsive, or do they appear jittery, delayed, or oddly synchronized with other facial features?",
    "Are there any unnatural warping or blending artifacts around the edges of the face, hair, or ears that suggest manipulation?",
    "Does the mouth shape and teeth structure appear consistent across frames, or are there visual glitches or unrealistic symmetry?",
    "Are facial expressions too static, identical across frames, or lacking micro-movements that would normally occur in real videos?"
    "Does the timing and shape of the mouth movements align naturally with the rest of the facial expression, or do they seem out of sync or detached?"
    "Is the facial geometry consistent across frames, or do features like the nose, jaw, or eyes shift or change proportion unnaturally?"
    "Do the edges of the hair or background exhibit flickering, smearing, or unnatural movement across frames?"
    "Do the pupils reflect light naturally and remain centered and consistent, or do they show flickering, deformation, or unnatural glare?"
    "Does the rotation and perspective of the head transition naturally between frames, or are there abrupt angle jumps or unnatural 3D distortions?"





]

INSTRUCTION = (
    "You are an expert in deepfake detection and forensic image analysis.\n"
    "Your task is to carefully analyze the provided image frame for any visual clues that suggest manipulation or inauthenticity.\n"
    "Pay special attention to unnatural textures, lighting mismatches, flickering patterns, overly smooth skin, eye/mouth artifacts, or rigid facial expressions.\n"
    "Respond ONLY in valid JSON format with:\n"
    "- \"label\": either \"real\" or \"deepfake\"\n"
    "- \"confidence\": number from 0.0 to 1.0\n"
    "- \"explanation\": a concise reason based on visual inconsistencies or forensic indicators"
)
'''
INSTRUCTION = (
    """
    You are an expert in deepfake detection and forensic image analysis.
    You are given one or several consecutive frames from a video.

    Your task is to carefully analyze ONLY the provided frames and decide if they look authentic or manipulated.
    Focus on forensic indicators such as:
    - lighting consistency (direction, shadows, background match),
    - skin texture (pores, wrinkles, over-smoothing),
    - eye and mouth dynamics (blinks, lip sync, teeth, tongue),
    - contour and blending (edges of hair/ears/face),
    - temporal artifacts (flickering, warping, ghosting),
    - head pose and geometry consistency.

    Respond STRICTLY in valid JSON format with exactly these keys:
    {
    "label": "real" or "deepfake",
    "confidence": float between 0.0 and 1.0,
    "explanation": "one short sentence with the main forensic reason"
    }
    """
)
PROMPTS = [
    # Lighting / shadows
    "Is the facial lighting direction and shadow placement consistent with the background and across frames?",
    # Skin texture
    "Does the skin show natural pores and wrinkles, or is it overly smooth, plastic-like, or inconsistent?",
    # Eye blinking
    "Are the blinks natural in frequency and duration, or do they appear robotic or absent?",
    # Eye gaze
    "Do the eyes track consistently and remain aligned without jitter or unnatural reflections?",
    # Mouth / lip sync
    "Do the lips, teeth, and tongue move naturally and in sync with expressions, without glitches?",
    # Contour / blending
    "Are the boundaries of face, hair, and ears clean and stable, without flicker or halo artifacts?",
    # Head pose / 3D geometry
    "Does the head pose transition smoothly without abrupt jumps or distorted proportions?",
    # Temporal artifacts
    "Do facial regions show flicker, smearing, or ghosting between frames?",
    # Expressions
    "Are expressions fluid and natural, or overly stiff and repetitive?"
]
