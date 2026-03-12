import json
from PIL import Image
from torch.utils.data import Dataset
import torch

from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# 1. 基础配置
# =========================
model_id = "llava-hf/llava-1.5-7b-hf"
data_path = "data/sft/train_sft.json"
output_dir = "outputs/sft_model"

print("1. 加载模型...")

# 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 加载量化模型
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 训练时关闭 cache
model.config.use_cache = False

# 为 k-bit 训练做准备
model = prepare_model_for_kbit_training(model)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# 挂载 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("2. 加载 processor...")
processor = AutoProcessor.from_pretrained(model_id)

# =========================
# 2. 数据集
# =========================
class MultimodalSFTDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")
        user_text = item["prompt"]
        assistant_text = item["response"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text},
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "image": image,
            "prompt": prompt,
        }

train_dataset = MultimodalSFTDataset(data_path)
print(f"3. 数据集加载完成，共 {len(train_dataset)} 条")

# =========================
# 3. 数据整理函数
# =========================
def collate_fn(batch):
    images = [example["image"] for example in batch]
    texts = [example["prompt"] for example in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    labels = inputs["input_ids"].clone()

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100

    inputs["labels"] = labels
    return inputs

# =========================
# 4. 训练参数
# =========================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=True,
    report_to="none",
)

# =========================
# 5. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

print("4. 开始训练...")
trainer.train()

print("5. 保存 LoRA adapter...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("训练完成。")