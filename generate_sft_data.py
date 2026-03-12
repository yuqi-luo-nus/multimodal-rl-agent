import json
import os

image_dir = "data/raw_images"
output_file = "data/sft/train_sft.json"

images = os.listdir(image_dir)

data = []

for img in images:
    item = {
        "id": img.split(".")[0],
        "image": f"data/raw_images/{img}",
        "prompt": "Look at the scientific experiment setup and output a logical step-by-step operation plan.",
        "response": "Step 1: Prepare the laboratory equipment shown in the image.\nStep 2: Arrange the instruments on the workbench and ensure they are properly connected.\nStep 3: Observe the readings or experimental indicators displayed on the devices.\nStep 4: Record the experimental results and clean the workspace."
    }
    
    data.append(item)

with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print("SFT dataset generated:", len(data))