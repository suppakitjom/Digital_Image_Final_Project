import face_recognition
import cv2
import os
import time

def extract_landmarks_and_write(image_folder, output_file_path, img_size=(128, 128)):
    start_time = time.time()
    num_samples = 0

    with open(output_file_path, 'w') as file:
        # Iterate through the '0' and '1' folders
        for smile_label in ['0', '1']:
            folder_path = os.path.join(image_folder, smile_label)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Load image
                image = face_recognition.load_image_file(image_path)

                # Resize image
                resized_image = cv2.resize(image, img_size)

                # Find face landmarks
                face_landmarks_list = face_recognition.face_landmarks(resized_image)

                # Skip images with no faces detected or more than one face
                if len(face_landmarks_list) != 1:
                    continue

                # Flatten the landmarks into a single list
                landmarks = [coord for landmark in face_landmarks_list[0].values() for coord in landmark]

                # Write landmarks and smile label to file
                file.write(f"{landmarks}, {smile_label}\n")
                num_samples += 1

    end_time = time.time()
    processing_time = end_time - start_time

    return processing_time, num_samples

# Paths
image_folder = 'lfw all images'  # Replace with the path to your images
output_file_path = 'data.txt'    # Replace with your desired output file path

# Extract landmarks and write to the file
processing_time, num_samples = extract_landmarks_and_write(image_folder, output_file_path)

print(f"Processing Time: {processing_time:.2f} seconds")
print(f"Number of Samples: {num_samples}")
