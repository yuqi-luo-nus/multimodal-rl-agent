from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

model_id = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

print("Model loaded successfully")