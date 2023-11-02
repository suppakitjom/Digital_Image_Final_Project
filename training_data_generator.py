import face_recognition
import numpy as np
import os
import cv2
import time

def process_image(file_path, img_size):
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        face_image = cv2.resize(face_image, img_size)
        face_landmarks_list = face_recognition.face_landmarks(face_image)

        if not face_landmarks_list:
            print(f"No landmarks found in {file_path}")

        blank_image = np.zeros((img_size[0], img_size[1], 1), np.uint8)
        for face_landmarks in face_landmarks_list:
            for feature_type in face_landmarks.keys():
                for point in face_landmarks[feature_type]:
                    cv2.circle(blank_image, point, 1, (255), -1)
        return blank_image
    else:
        print(f"No face detected in {file_path}")
        return np.zeros((img_size[0], img_size[1], 1), np.uint8)

def generate_data(directory, img_size=(128, 128)):
    start_time = time.time()
    labels = {'0': 0, '1': 1}
    output_file = open('output_data.txt', 'w')
    data_count = 0

    for label, label_value in labels.items():
        folder_path = os.path.join(directory, label)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    plotted_image = process_image(file_path, img_size)
                    if np.any(plotted_image):  # Check if the image is not entirely zero
                        output_file.write(f"{plotted_image.tolist()},{label_value}\n")
                        data_count += 1
                    else:
                        print(f"Blank image for {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    output_file.close()
    end_time = time.time()
    print(f"Data generation completed in {end_time - start_time:.2f} seconds.")
    print(f"Total number of data points: {data_count}")

generate_data('lfw all images')
