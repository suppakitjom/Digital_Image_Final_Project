# Automatic Selfie Taker using Smile Detection

Authors:

Suppakit Laomahamek 6438232221

Bhammanas Praesangeim 6438173921

Tibet Buramarn 6438108121

If you've ever used a Samsung mobile device, you've probably tried taking a selfie by showing your palm. Our feature detects smiles instead - a photo is taken when everyone in the frame smiles.

The process of training a model to be able to detect whether a person is smiling is simple:

1. Label images whether the person in the image is smiling or not

1. Find faces in each image

1. Resize facial images into 128x128 images

1. Detect facial landmarks and plot landmarks into a blank picture, correcting the face alignment by rotating the image until the nose is vertical

1. Train model using the image of plotted landmarks and their label

![Original Image](https://cdn-images-1.medium.com/max/2000/1*YvvduWTJTPT5PL2QPfm8mg.png)

_Original Image_

![Cropped Image of Detected Face](https://cdn-images-1.medium.com/max/2000/1*vZDTNeQ6gfWbIoAJGgVTvw.png)

_Cropped Image of Detected Face_

![Facial Landmark of Detected Face](https://cdn-images-1.medium.com/max/2000/1*LXEIUmf0ykYAeDi9sbucYA.png)

_Facial Landmark of Detected Face_

![Angle Corrected Facial Landmark Image](https://cdn-images-1.medium.com/max/2000/1*Der8Fwwkrk046BSs6Gk8rw.png)

_Angle Corrected Facial Landmark Image_

**Results**

Our dataset contains approximately 500 images, so the overall accuracy might be relatively low compared to pre-trained models with larger datasets and resources.

Confusion Matrix

![](https://cdn-images-1.medium.com/max/2000/1*i1dDA46szKmSL0G9sK7B2A.png)

Accuracy = 0.26, Precision = 0.2143, Recall = 0.18, F1-score = 0.1957

Inference Time = 12 ms

## **Comparison with Haar Cascade Method**

### Overview of Haar Cascade

The Haar Cascade method involves the OpenCV cascade classifier, an implementation of the Haar Cascade algorithm used for object detection in images and videos. This classifier works by sliding a window over the input image at various scales, identifying regions that match the learned features of the target object, and rejecting regions that do not.

Different XML files for cascade classifiers are applied to identify the most effective one. The file “Haarcascade_frontalface_default.xml” is used for face detection, while “Haarcascade_smile.xml” is utilized for smile detection. Additionally, YoloV8 is employed to assist in counting the total number of faces in the image.

The condition under which an image is marked as positive is that the number of smiles detected must equal the total number of faces identified in the image.

Test Set Code

    from ultralytics import YOLO
    import cv2
    import os

    # face_cascade = cv2.CascadeClassifier('default_frontal_face.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # smile_cascade = cv2.CascadeClassifier('default_smile_cascade.xml')
    # smile_cascade = cv2.CascadeClassifier('smile_cascade.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    yolo_face = 0
    cascade_face = 0
    cascade_smile = 0

    def detect_smiles(gray, frame):
        global frame_cop
        frame_cop = frame.copy()
        global a
        global b
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        a = len(faces)
        b = 0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.9, 20)
            if len(smiles) != 0:
                b += 1
            cv2.rectangle(frame_cop, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame_cop, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

        return frame_cop, a, b

    CONFIDENCE_THRESHOLD = 0.2
    GREEN = (0, 255, 0)
    THICKNESS = -1

    model = YOLO("yolov8n-face.pt")

    input_folder = 'class0'
    output_folder = 'res0'


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = os.listdir(input_folder)
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        print(image_path)
        frame = cv2.imread(image_path)
        detections = model(frame)[0]
        img = frame.copy()

        yolo_face = len(detections.boxes.data.tolist())
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), GREEN, 4)

        cv2.imwrite(output_path, img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_with_boxes, cascade_face, cascade_smile = detect_smiles(gray, frame.copy())
        print(yolo_face, cascade_face, cascade_smile)
        if (yolo_face == cascade_smile):
            output_class_folder = os.path.join(output_folder, 'class1')
        else:
            output_class_folder = os.path.join(output_folder, 'class0')

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        output_path = os.path.join(output_class_folder, image_file)

        cv2.imwrite(output_path, frame_with_boxes)
        print(f"Processed {image_file} and saved to {output_class_folder}")

    input_folder = 'class1'
    output_folder = 'res1'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = os.listdir(input_folder)
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        print(image_path)
        frame = cv2.imread(image_path)
        detections = model(frame)[0]
        img = frame.copy()

        yolo_face = len(detections.boxes.data.tolist())
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), GREEN, 4)

        cv2.imwrite(output_path, img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_with_boxes, cascade_face, cascade_smile = detect_smiles(gray, frame.copy())
        print(yolo_face, cascade_face, cascade_smile)
        if (yolo_face == cascade_smile):
            output_class_folder = os.path.join(output_folder, 'class1')
        else:
            output_class_folder = os.path.join(output_folder, 'class0')

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        output_path = os.path.join(output_class_folder, image_file)

        cv2.imwrite(output_path, frame_with_boxes)
        print(f"Processed {image_file} and saved to {output_class_folder}")

    TN = len(os.listdir('res0/class0'))
    FP = len(os.listdir('res0/class1'))
    FN = len(os.listdir('res1/class0'))
    TP = len(os.listdir('res1/class1'))

    print('----------------------------------------------')
    # print(TN, FP, FN, TP)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1s = (2 * precision * recall) / (precision + recall)
    print('TP: '+str(TP)+' FN: '+str(FN)+' FP: '+str(FP)+' TN: '+str(TN))

    print('Accuracy = '+str(accuracy))
    print('Precision = '+str(precision))
    print('Recall = '+str(recall))
    print('F1 = '+str(F1s))

Live Camera Code

    from ultralytics import YOLO
    import cv2

    # face_cascade = cv2.CascadeClassifier('default_frontal_face.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # smile_cascade = cv2.CascadeClassifier('default_smile_cascade.xml')
    # smile_cascade = cv2.CascadeClassifier('smile_cascade.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    CONFIDENCE_THRESHOLD = 0.4
    GREEN = (0, 255, 0)
    THICKNESS = -1
    video_cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n-face.pt")

    yolo_face = 0
    cascade_face = 0
    cascade_smile = 0

    def detect_smiles(gray, frame):
        global frame_cop
        frame_cop = frame.copy()
        global a
        global b
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        a = len(faces)
        b = 0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.9, 20)
            if len(smiles) != 0:
                b += 1
            cv2.rectangle(frame_cop, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame_cop, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

        return frame_cop, a, b


    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        detections = model(frame)[0]

        img = frame.copy()

        yolo_face = len(detections.boxes.data.tolist())
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), GREEN, 4)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_with_boxes, cascade_face, cascade_smile = detect_smiles(gray, frame.copy())
        cv2.imshow('Video', frame_with_boxes)
        print(yolo_face, cascade_face, cascade_smile)
        if (yolo_face == cascade_smile):
            cv2.imwrite('smiling_faces_image.jpg', frame)
            break
        if cv2.waitKey(1) == ord("q"):
            break

    video_cap.release()
    cv2.destroyAllWindows()

**Source**: [\*https://github.com/bhambhambhambham/DIP_Comp_Model](https://github.com/bhambhambhambham/DIP_Comp_Model)\*

**Results**

![Smiles detected using the Haar Cascade Model](https://cdn-images-1.medium.com/max/2400/0*g5BaKRv7f5iaOHIj)

_Smiles detected using the Haar Cascade Model_

Confusion Matrix

![](https://cdn-images-1.medium.com/max/2000/1*mG-gxKrw1D8d8rDFG-dKmw.png)

Accuracy = 0.53, Precision = 0.6, Recall = 0.18, F1-score = 0.2769

Inference Time = 111.1 ms

**Results Comparison**

In comparing the performance of our model and the Haar Cascade model in object detection tasks, we can analyze their effectiveness based on several key metrics: Accuracy, Precision, Recall, F1-score, and Inference Time.

1. **_Accuracy:_**

_Our Model: 0.26_

_Haar Cascade Model: 0.53_

**Observation:** The Haar Cascade model shows more than double the accuracy of our model, indicating it is more reliable in correctly identifying targets.

2. **_Precision:_**

*Our Model: *0.2143

*Haar Cascade Model: *0.6

**Observation:** The Haar Cascade model has significantly higher precision, implying that when it predicts a target, it is more likely to be correct compared to our model.

3. **_Recall:_**

*Our Model: *0.18

_Haar Cascade Model_: 0.18

**Observation:** Both models have the same recall rate, suggesting they are equally capable of identifying all relevant instances in the dataset.

4. **_F1-score:_**

*Our Model: *0.1957

*Haar Cascade Model: *0.2769

**Observation: **The Haar Cascade model achieves a higher F1-score, indicating a better balance between precision and recall compared to our model.

5. **_Inference Time:_**

*Our Model: *12 ms

_Haar Cascade Model:_ 111.1 ms

**Observation:** Our model is significantly faster in inference, making it more suitable for applications where speed is critical.

**Final Thoughts
**The Haar Cascade Model outperforms our model in terms of accuracy, precision, and F1-score, making it more effective in correctly identifying and classifying smiling faces. However, its significantly longer inference time suggests a trade-off between accuracy and speed. Our model, while less accurate, offers much faster processing, crucial for real-time applications like a smile detector where prompt response is important. The choice between these models would depend on whether prioritizing the accuracy of detecting smiles or ensuring quicker image capture is more critical for the project.

**Pros** : The main advantage of CNN is that it can learn from raw pixel data without requiring any manual feature engineering and is adaptable to unseen data.

**Cons** : CNN requires a large amount ox`f labeled data to train effectively. And is prone to overfitting, which means that they can memorize the noise and details of the training data and fail to generalize to new and different data.
