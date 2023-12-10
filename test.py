import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
from PIL import Image
import argparse
# import keyboard
from time import time,sleep

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
DEBUG = args.debug

# Load the trained model
model = load_model('model2.keras') 

def take_photo(video_capture):
    start_time = time()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop if unable to capture a frame

        elapsed_time = time() - start_time
        remaining_time = 5 - int(elapsed_time)
        cv2.putText(frame, str(max(remaining_time, 0)), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 10, cv2.LINE_AA)
        
        cv2.imshow('Video', frame)
        
        if elapsed_time >= 5:
            ret, frame = video_capture.read()
            cv2.imwrite('captured_image.jpg', frame)
            print("Image captured and saved as 'captured_image.jpg'")
            break  # Break the loop after the photo is taken

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def process_image(face_image):
    IMG_SIZE = (128, 128)
    face_image = cv2.resize(face_image, IMG_SIZE)

    # Get facial landmarks
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    face_landmarks = face_landmarks_list[0] if face_landmarks_list else None

    if not face_landmarks:
        raise ValueError("No facial landmarks were detected.")

    # Create a blank image with the same size as the face_image
    blank_image = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), np.uint8)

    # Plot landmarks on the blank image
    for feature in face_landmarks:
        for point in face_landmarks[feature]:
            x, y = point
            x = min(max(0, x), IMG_SIZE[0] - 1)
            y = min(max(0, y), IMG_SIZE[1] - 1)
            blank_image[y, x] = 255

    # Get the nose bridge points to calculate the rotation angle
    top_nose_bridge = face_landmarks['nose_bridge'][0]
    bottom_nose_bridge = face_landmarks['nose_bridge'][-1]
    dY = bottom_nose_bridge[1] - top_nose_bridge[1]
    dX = bottom_nose_bridge[0] - top_nose_bridge[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 90

    # Rotate the blank image with landmarks
    img = Image.fromarray(blank_image.astype('uint8').squeeze(), 'L')
    rotated_img = img.rotate(angle)
    resized_rotated_img = rotated_img.resize(IMG_SIZE)
    rotated_image_array = np.array(resized_rotated_img)
    rotated_image_array = rotated_image_array.reshape((IMG_SIZE[0], IMG_SIZE[1], 1))

    return rotated_image_array

video_capture = cv2.VideoCapture(0)

while True:
    # Initialize the smile counter for each frame
    smile_count = 0
    people_count = 0

    ret, frame = video_capture.read()
    
    if not ret:
        break  # If no frame is captured, break the loop

    # Resize frame to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_locations = [(top*4, right*4, bottom*4, left*4) for top, right, bottom, left in face_locations]

    for face_location in face_locations:
        people_count += 1

        # Extract the face image
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]

        # Process the face image
        try:
            processed_face = process_image(face_image)
            processed_face = np.expand_dims(processed_face, axis=0)
            processed_face = processed_face / 255.0

            # Use the model to predict if the person is smiling
            prediction = model.predict(processed_face)
            smile_prob = prediction[0][1]
            if DEBUG:
                print(prediction[0][1])
            
            if smile_prob > 1e-1:  # Adjust the threshold as needed
                smile_count += 1
                cv2.putText(frame, 'Smiling :)', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Not Smiling', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        except Exception as e:
            print(f"Error in processing the image: {e}")

    # Display the smiling count on the frame
    smile_text = f"Smiling: {smile_count}/{people_count}"

    if smile_count == people_count and (smile_count != 0):
        take_photo(video_capture)

    cv2.putText(frame, smile_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        print('taking photo')
        take_photo(video_capture)

video_capture.release()
cv2.destroyAllWindows()
