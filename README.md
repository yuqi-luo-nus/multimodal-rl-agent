# Multimodal Reasoning Agent  
LLaVA + LoRA Fine-tuning for Step-by-Step Task Planning

## Overview

This project implements a **multimodal reasoning agent** capable of generating **step-by-step operational plans** from images of scientific setups.

The system is built on top of **LLaVA (Large Language and Vision Assistant)** and fine-tuned using **LoRA (Low-Rank Adaptation)** with **4-bit quantization** for efficient training on a single GPU.

The goal of this project is to demonstrate how multimodal foundation models can be adapted for **visual reasoning and task planning**.

Example task:

> Input: Image of an experimental setup  
> Output: Logical step-by-step instructions for performing the experiment.

---

## Features

- Multimodal reasoning (image + language)
- LLaVA 1.5 base model
- LoRA parameter-efficient fine-tuning
- 4-bit quantization for low GPU memory usage
- HuggingFace Transformers integration
- End-to-end training and inference pipeline

---

## Project Structure


multimodal-rl-agent
│
├── data
│ ├── dataset.json
│ └── raw_images/
│
├── train_sft.py # LoRA fine-tuning script
├── test_lora.py # inference with LoRA adapter
├── inference.py # simple demo inference script
│
├── outputs
│ └── sft_model # saved LoRA adapters
│
├── requirements.txt
└── README.md


---

## Model Architecture

Base model:


LLaVA-1.5-7B


Fine-tuning method:


LoRA (Low-Rank Adaptation)


Quantization:


4-bit (bitsandbytes)


Frameworks:

- PyTorch
- HuggingFace Transformers
- PEFT
- Accelerate

---

## Installation

Clone the repository:


git clone https://github.com/yourusername/multimodal-rl-agent.git

cd multimodal-rl-agent


Create virtual environment:


python -m venv venv
venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


---

## Training

Run LoRA fine-tuning:


python train_sft.py


Training pipeline:


Dataset
↓
LLaVA Base Model
↓
4-bit Quantization
↓
LoRA Fine-tuning
↓
Save Adapter


The trained LoRA adapter will be saved to:


outputs/sft_model


---

## Inference

Run the test script:


python test_lora.py


Example prompt:


Look at this scientific experiment setup and produce a logical step-by-step operation plan.


Example output:


Step 1: Prepare the experiment setup.
Step 2: Arrange the materials and equipment.
Step 3: Position the components correctly.
Step 4: Execute the experimental procedure.
Step 5: Record and analyze the results.


---

## Example Workflow


Image
↓
LLaVA Vision Encoder
↓
Language Model
↓
LoRA Adaptation
↓
Step-by-step reasoning


---

## Hardware Requirements

Recommended:


GPU: RTX 3060 / 3070 / 4060 / 4090
VRAM: ≥ 8GB


Training with 4-bit quantization allows running large multimodal models on consumer GPUs.

---

## Future Improvements

Possible next steps:

- Reinforcement Learning (RLHF / RLAIF)
- Larger datasets
- Multi-image reasoning
- Tool use and agent planning
- Deployment as an interactive AI assistant

---

## Acknowledgements

This project builds on the following open-source work:

- LLaVA: https://github.com/haotian-liu/LLaVA
- HuggingFace Transformers
- PEFT (Parameter Efficient Fine-Tuning)

---

## License

MIT License
