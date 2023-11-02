import face_recognition
import cv2

# Start the webcam capture
video_capture = cv2.VideoCapture(0)

process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()


    # Only process every other frame of video to save time
    if process_this_frame:
        faces = []
        face_landmarks = []
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # for each face location, generate an image that contains only that face that is 128x128 big
        for face_location in face_locations:
            # Scale back up face locations
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Extract the region of the image that contains the face
            face_image = frame[top:bottom, left:right]
            
            # Resize the face image to 128x128
            face_image = cv2.resize(face_image, (128, 128))
            faces.append(face_image)
            
            # Find landmarks for each face
            face_landmark = face_recognition.face_landmarks(face_image)
            face_landmarks.append(face_landmark)
            print(face_landmark,'\n')

        # Find face landmarks
        face_landmarks = face_recognition.face_landmarks(rgb_small_frame)

    # Draw face landmarks
    for face_landmark in face_landmarks:
        for facial_feature in face_landmark.keys():
            for point in face_landmark[facial_feature]:
                # Scale back up face landmarks points
                scaled_point = (point[0] * 4, point[1] * 4)
                cv2.circle(frame, scaled_point, 5, (0, 255, 0), -1)

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
