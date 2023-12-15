import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
from PIL import Image
import argparse
from time import time,sleep
import os

model = load_model('model2.keras') 

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


# variables for the confusion matrix
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

# load each picture in folder named 0
for filename in os.listdir('./testing image/0'):
    # variable to check if all faces in the image are not smiling
    all_faces_not_smiling = True
    # load image
    image = face_recognition.load_image_file('./testing image/0/' + filename)
    # get face locations
    face_locations = face_recognition.face_locations(image)
    # for each face image, call the process_image function
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        try:
            processed_image = process_image(face_image)
            processed_image = np.expand_dims(processed_image, axis=0)
            prediction = model.predict(processed_image)
            print(prediction)
            if np.argmax(prediction[0]) != 0:
                all_faces_not_smiling = False
        except ValueError as e:
            print(e)
            continue
    if all_faces_not_smiling:
        true_negative += 1
    else:
        false_positive += 1

# load each picture in folder named 1
for filename in os.listdir('./testing image/1'):
    all_faces_smiling = True
    # load image
    image = face_recognition.load_image_file('./testing image/1/' + filename)
    # get face locations
    face_locations = face_recognition.face_locations(image)
    # for each face image, call the process_image function
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        try:
            processed_image = process_image(face_image)
            processed_image = np.expand_dims(processed_image, axis=0)
            prediction = model.predict(processed_image)
            print(prediction)
            if np.argmax(prediction[0]) != 1:
                all_faces_smiling = False
        except ValueError as e:
            print(e)
            continue
    if all_faces_smiling:
        true_positive += 1
    else:
        false_negative += 1

# print TP FN TN FP
print(f"TP:{true_positive} FN:{false_negative} TN:{true_negative} FP:{false_positive}")

print(f"Accuracy: {(true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)}")
print(f"Precision: {true_positive / (true_positive + false_positive)}")
print(f"Recall: {true_positive / (true_positive + false_negative)}")
print(f"F1 Score: {(2 * true_positive) / (2 * true_positive + false_positive + false_negative)}")