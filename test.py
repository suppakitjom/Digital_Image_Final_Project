import cv2
import face_recognition
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('smiling_detection_model.h5')  # Replace with the path to your saved model

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Initialize scaler (approximating the original scaler)
scaler = StandardScaler()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = small_frame[:, :, ::-1]

    # Find all face locations in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    for face_location in face_locations:
        # Extract the region of interest (the face)
        top, right, bottom, left = face_location
        face_image = rgb_frame[top:bottom, left:right]

        # Resize face image to 128x128
        face_image_resized = cv2.resize(face_image, (128, 128))

        # Find face landmarks
        face_landmarks_list = face_recognition.face_landmarks(face_image_resized)

        # Proceed only if landmarks are detected
        if face_landmarks_list:
            # Flatten the landmarks
            landmarks = [coord for landmark in face_landmarks_list[0].values() for coord in landmark]
            landmarks = np.array([item for sublist in landmarks for item in sublist]).reshape(1, -1)

            # Normalize the landmarks
            landmarks_normalized = scaler.fit_transform(landmarks)

            # Predict smiling
            prediction = model.predict(landmarks_normalized)
            is_smiling = prediction[0][0] > 1e-22

            # Print the prediction value for debugging
            print("Prediction:", prediction)

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Display the results
            color = (0, 255, 0) if is_smiling else (0, 0, 255)
            label = "Smiling" if is_smiling else "Not Smiling"

            # Draw a box around the face and label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
