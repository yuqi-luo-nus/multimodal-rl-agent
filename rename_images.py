import os

folder = "data/raw_images"

files = os.listdir(folder)

i = 1
for f in files:
    if f.endswith(".jpg") or f.endswith(".png"):
        new_name = f"img_{i:03d}.jpg"
        os.rename(
            os.path.join(folder, f),
            os.path.join(folder, new_name)
        )
        i += 1

print("rename done")