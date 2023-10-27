import cv2
import os
from pathlib import Path
from shutil import move

# Directory containing the images
image_dir = 'lfw all images'  # Update this path
folders = {'1': '1', '0': '0'}

# Create target folders if they don't exist
for folder in folders.values():
    Path(os.path.join(image_dir, folder)).mkdir(exist_ok=True)

# Iterate over each image
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    
    # Skip if it's a directory
    if os.path.isdir(image_path):
        continue

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Display the image
    cv2.imshow('Image', img)
    
    # Wait for the key press
    key = cv2.waitKey(0) & 0xFF

    # Move the image to the corresponding folder based on key press
    if chr(key) in folders:
        new_path = os.path.join(image_dir, folders[chr(key)], image_name)
        move(image_path, new_path)
    
    # Break the loop if 'esc' key is pressed
    if key == 27:
        break

cv2.destroyAllWindows()