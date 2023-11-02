import face_recognition
import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('my_cnn_model.h5')

# Function to process face images
def process_face_image(face_image):
    # Resize face image to 128x128 and grayscale as per the model's training data
    face_image_resized = cv2.resize(face_image, (128, 128))
    face_image_resized = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2GRAY)
    face_image_resized = np.expand_dims(face_image_resized, axis=-1)
    face_image_resized = np.expand_dims(face_image_resized, axis=0)
    face_image_resized = face_image_resized / 255.0  # Normalize the image
    return face_image_resized

# Start the webcam capture
video_capture = cv2.VideoCapture(0)

process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face landmarks in the small frame
        face_locations = face_recognition.face_locations(small_frame)
        for face_location in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Extract the face
            face_image = frame[top:bottom, left:right]

            # Process the face image
            processed_face = process_face_image(face_image)

            # Predict if the person is smiling
            prediction = model.predict(processed_face)
            if prediction[0][0] < 0.395:  # Adjust threshold as per your model's performance
                label = "Smiling"
                frame_color = (0, 255, 0)  # Green frame for smiling
            else:
                label = "Not Smiling"
                frame_color = (0, 0, 255)  # Red frame for not smiling
            print(prediction,label)


            # Display label on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), frame_color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_color, 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
