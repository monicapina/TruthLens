# prompts_bank.py
from typing import Dict, List

ATTRIBUTES: Dict[str, Dict[str, List[str]]] = {
    "lighting_consistency": {
        "suspicious": [
            "lighting inconsistency between face and background",
            "inconsistent shadow direction on the face",
            "mismatched highlights and shadows across face and scene",
        ],
        "realistic": [
            "consistent lighting on face and background",
            "shadows and highlights align with scene lighting",
        ],
    },
    "skin_texture": {
        "suspicious": [
            "unnatural skin texture",
            "waxy or overly smooth facial texture",
            "texture inconsistency between facial regions",
        ],
        "realistic": [
            "natural skin texture with fine detail",
            "consistent facial texture across regions",
        ],
    },
    "eye_blinking": {
        "suspicious": ["unnatural eye blinking pattern", "irregular or absent eye blinks"],
        "realistic": ["natural and regular eye blinking"],
    },
    "lip_motion_consistency": {
        "suspicious": ["lip motion inconsistent with speech", "asynchronous lip movements"],
        "realistic": ["lip motion consistent with speech"],
    },
    "contour_blending": {
        "suspicious": ["blurred or inconsistent face contour blending", "halo artifacts near facial boundary"],
        "realistic": ["clean and consistent facial boundary"],
    },
    "pose_3d": {
        "suspicious": ["inconsistent 3D head pose with background", "impossible head pose transitions"],
        "realistic": ["head pose consistent with scene and camera"],
    },
    "temporal_artifacts": {
        "suspicious": ["temporal flickering across frames", "inconsistent details between consecutive frames"],
        "realistic": ["temporal consistency across frames"],
    },
    "face_background_separation": {
        "suspicious": ["unnatural separation between face and background", "depth or matting inconsistency around face"],
        "realistic": ["natural separation between face and background"],
    },
    "accessories_consistency": {
        "suspicious": ["inconsistent rendering of accessories such as hats or glasses"],
        "realistic": ["accessories rendered consistently with lighting and pose"],
    },
    "expressive_dynamics": {
        "suspicious": ["unnatural facial expression dynamics", "rigid or mismatched expressions over time"],
        "realistic": ["natural facial expression dynamics"],
    },
}

CLASS_PROMPTS = {
    "real": [
        "this is a real, unmanipulated human face image",
        "authentic human face without manipulation",
    ],
    "deepfake": [
        "this is a manipulated deepfake human face image",
        "human face generated or altered by deepfake techniques",
    ],
}
