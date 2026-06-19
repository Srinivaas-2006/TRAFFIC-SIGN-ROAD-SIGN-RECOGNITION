!pip install -q kagglehub
import kagglehub
import os, random
import matplotlib.pyplot as plt
from PIL import Image
path = kagglehub.dataset_download(
    "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
)
images = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith((".png",".jpg",".ppm")):
            images.append(os.path.join(root,f))

print("Total images available:", len(images))
plt.figure(figsize=(15,20))

for i in range(50):
    img_path = random.choice(images)
    img = Image.open(img_path)

    plt.subplot(10,5,i+1)
    plt.imshow(img)
    plt.axis("off")

plt.tight_layout()
plt.show()
