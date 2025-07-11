import os
import cv2
from skimage.util import random_noise

cover_folder = 'data/cover'
stego_folder = 'data/stego'

os.makedirs(stego_folder, exist_ok=True)

for file in os.listdir(cover_folder):
    path = os.path.join(cover_folder, file)
    img = cv2.imread(path)
    if img is None:
        continue
    # Add random noise (to simulate stego)
    noisy_img = random_noise(img, mode='s&p', amount=0.05)
    noisy_img = (255 * noisy_img).astype('uint8')
    cv2.imwrite(os.path.join(stego_folder, file), noisy_img)

print("✅ Stego images created in data/stego/")
