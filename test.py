import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('model.keras')  # Replace with your model path

# Define the process_image function you provided
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
            # Ensure x and y are within the bounds of the image size
            x = min(max(0, x), IMG_SIZE[0] - 1)
            y = min(max(0, y), IMG_SIZE[1] - 1)
            blank_image[y, x] = 255  # white color for the landmark points

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

# Open a connection to the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        break  # If no frame is captured, break the loop

    # Resize frame to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Now we need to scale back up face locations since the frame we detected in was scaled to 1/4 size
    face_locations = [(top*4, right*4, bottom*4, left*4) for top, right, bottom, left in face_locations]

    for face_location in face_locations:
        # Extract the face image
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]  # Note: Using the original frame, not the scaled one

        # Process the face image
        try:
            processed_face = process_image(face_image)
            processed_face = np.expand_dims(processed_face, axis=0)  # Add batch dimension
            processed_face = processed_face / 255.0  # Normalize the image

            # Use the model to predict if the person is smiling
            prediction = model.predict(processed_face)
            print(prediction)
            smile_prob = prediction[0][1]
            
            # Display the result on the frame
            if smile_prob > 1e-4:  # You can adjust this threshold
                cv2.putText(frame, 'Smiling :)', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Not Smiling', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        except Exception as e:
            print(f"Error in processing the image: {e}")

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
