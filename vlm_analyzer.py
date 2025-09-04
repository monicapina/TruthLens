import torch
import json
import re
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor, BitsAndBytesConfig

class DeepfakeVLMAnalyzer:
    def __init__(self, model_name="OpenGVLab/InternVL2_5-8B", device="cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {self.device}")
        self._load_model()

    def _load_model(self):
        print("\n📦 Loading InternVL2_5-8B...")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name)

    def query(self, images, prompt):
        pixel_values = self.image_processor(images=images, return_tensors="pt").pixel_values.to(torch.bfloat16).to(self.device)
        generation_config = dict(max_new_tokens=512, do_sample=False)
        try:
            response = self.model.chat(self.tokenizer, pixel_values=pixel_values, question=prompt, generation_config=generation_config)
            match = re.search(r"\{.*\}", response, re.DOTALL)
            return json.loads(match.group(0)) if match else {"label": "error", "confidence": None, "explanation": response}
        except Exception as e:
            return {"label": "error", "confidence": None, "explanation": str(e)}
