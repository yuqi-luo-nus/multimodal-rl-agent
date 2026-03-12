from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

print("1. 开始加载模型...")

model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("2. 模型加载完成")

processor = AutoProcessor.from_pretrained(model_id)
print("3. processor 加载完成")
image = Image.open("data/raw_images/img_001.jpg").convert("RGB")
# image = Image.open("data/raw_images/000001.jpg").convert("RGB")
print("4. 图片读取完成")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "You are a laboratory assistant. "
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

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print("5. prompt 构造完成")

inputs = processor(images=image, text=prompt, return_tensors="pt")

inputs = {k: v.to(model.device) for k, v in inputs.items()}
print("6. 输入处理完成，开始生成...")

output = model.generate(**inputs, max_new_tokens=150)

print("7. 生成完成，下面是输出：")
print(processor.decode(output[0], skip_special_tokens=True))