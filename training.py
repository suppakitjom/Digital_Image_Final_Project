import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Load and process the data using Pandas
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, converters={0: lambda x: np.fromstring(x.strip('[]'), sep=', ')})
    images = np.stack(df[0].values)
    labels = df[1].values
    return images, labels

# Load data
images, labels = load_data('output_data.txt')

# Preprocess data
images = images.reshape(-1, 128, 128, 1)  # Reshape for CNN
images = images / 255.0  # Normalize pixel values

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)

# Model summary
model.summary()

# Save the model
model.save('my_cnn_model.h5')
