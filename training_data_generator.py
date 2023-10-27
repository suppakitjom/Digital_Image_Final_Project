import face_recognition
import os

def extract_landmarks_and_write(image_folder, output_file_path):
    with open(output_file_path, 'w') as file:
        # Iterate through the '0' and '1' folders
        for smile_label in ['0', '1']:
            folder_path = os.path.join(image_folder, smile_label)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Load image and find face landmarks
                image = face_recognition.load_image_file(image_path)
                face_landmarks_list = face_recognition.face_landmarks(image)

                # Skip images with no faces detected or more than one face
                if len(face_landmarks_list) != 1:
                    continue

                # Flatten the landmarks into a single list
                landmarks = [coord for landmark in face_landmarks_list[0].values() for coord in landmark]

                # Write landmarks and smile label to file
                file.write(f"{landmarks}, {smile_label}\n")

# Paths
image_folder = 'lfw all images'  # Replace with the path to your images
output_file_path = 'data.txt'    # Replace with your desired output file path

# Extract landmarks and write to the file
extract_landmarks_and_write(image_folder, output_file_path)
