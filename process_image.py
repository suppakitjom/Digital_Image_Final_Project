import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time
from skimage.feature import hog

PATH = 'training image'
def process_image(face_image):
    # Resize image to 128x128 if it's not already
    IMG_SIZE = (128, 128)
    if face_image.shape[:2] != IMG_SIZE:
        face_image = cv2.resize(face_image, IMG_SIZE)
    
    # Get facial landmarks
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    
    # For simplicity, we take the first set of landmarks
    face_landmarks = face_landmarks_list[0] if face_landmarks_list else None
    
    if not face_landmarks:
        raise ValueError("No facial landmarks were detected.")

    # Create a blank image with the same size as the face_image
    blank_image = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), np.uint8)

    # Plot landmarks on the blank image
    for feature in face_landmarks:
        for point in face_landmarks[feature]:
            x, y = point
            if 0 <= x < IMG_SIZE[0] and 0 <= y < IMG_SIZE[1]:
                blank_image[y, x] = 255  # white color for the landmark points

    # Get the nose bridge points to calculate the rotation angle
    if 'nose_bridge' in face_landmarks:
        top_nose_bridge = face_landmarks['nose_bridge'][0]
        bottom_nose_bridge = face_landmarks['nose_bridge'][-1]
        dY = bottom_nose_bridge[1] - top_nose_bridge[1]
        dX = bottom_nose_bridge[0] - top_nose_bridge[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 90
    else:
        raise ValueError("Nose bridge landmarks are required for rotation calculation but were not found.")

    # Rotate the blank image with landmarks
    img = Image.fromarray(blank_image.astype('uint8').squeeze(), 'L')
    rotated_img = img.rotate(angle)
    resized_rotated_img = rotated_img.resize(IMG_SIZE)
    rotated_image_array = np.array(resized_rotated_img)
    rotated_image_array = rotated_image_array.reshape((IMG_SIZE[0], IMG_SIZE[1], 1))
    
    return rotated_image_array


def generate_data(directory, img_size=(128, 128)):
    start_time = time.time()
    labels = {'0': 0, '1': 1}
    data_count = 0

    processed_img_dir = os.path.join(directory, 'processed_img')
    if not os.path.exists(processed_img_dir):
        os.makedirs(processed_img_dir, exist_ok=True)

    for label, label_value in labels.items():
        folder_path = os.path.join(directory, label)
        # create output folder
        output_folder_path = os.path.join(processed_img_dir, label)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    # load img
                    image = face_recognition.load_image_file(file_path)
                    # get face locations
                    face_locations = face_recognition.face_locations(image)
                    # for each face image, call the process_image function
                    for face_location in face_locations:
                        top, right, bottom, left = face_location
                        face_image = image[top:bottom, left:right]
                        rotated_image_array = process_image(face_image)
                        # save the image to the output folder
                        output_file_path = os.path.join(output_folder_path, filename)
                        print(f"Saving {output_file_path}")
                        cv2.imwrite(output_file_path, rotated_image_array)
                        data_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    end_time = time.time()
    print(f"Data generation completed in {end_time - start_time:.2f} seconds.")
    print(f"Total number of data points: {data_count}")
generate_data(PATH)