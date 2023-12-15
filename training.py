import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Function to load images from the directory
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if not os.path.isdir(label_folder):  # Check if it is a directory
            continue  # Skip files like .DS_Store
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # assuming the images are in grayscale
            if img is not None:
                images.append(img)
                labels.append(int(label))
    return np.array(images), np.array(labels)


# Load images
images, labels = load_images_from_folder('./lfw all images/processed_img')

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# Resize images to the expected size (128x128x1 for grayscale)
images = np.expand_dims(images, axis=-1)

# Convert the labels to one-hot encoding
labels = to_categorical(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax') # 2 classes: smiling or not smiling
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
model.save('modellll.keras')
print(f"Test accuracy: {test_acc:.4f}")
