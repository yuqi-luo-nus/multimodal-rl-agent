import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "llava-hf/llava-1.5-7b-hf"
adapter_path = "outputs/sft_model"

print("1. 加载 4bit base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base_model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("2. 加载 LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("3. 加载 processor...")
processor = AutoProcessor.from_pretrained(base_model_id)

print("4. 读取图片...")
image = Image.open("data/raw_images/img_001.jpg").convert("RGB")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "Look at this scientific experiment setup and produce a logical "
                    "step-by-step operation plan.\n\n"
                    "Output format:\n"
                    "Step 1: ...\n"
                    "Step 2: ...\n"
                    "Step 3: ...\n"
                    "Step 4: ..."
                ),
            },
        ],
    }
]

prompt = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False,
)

inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt",
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("5. 开始推理...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
    )

print("\n===== 模型输出 =====")
print(processor.decode(output[0], skip_special_tokens=True))