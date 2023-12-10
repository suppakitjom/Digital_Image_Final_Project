    import os
    import cv2
    import numpy as np
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Function to load images from the directory
    def load_images_from_folder(folder):
        images = []
        labels = []
        for label in os.listdir(folder):
            label_folder = os.path.join(folder, label)
            if not os.path.isdir(label_folder):
                continue
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(int(label))
        return np.array(images), np.array(labels)

    # Load images
    images, labels = load_images_from_folder('./lfw all images/processed_img')

    # Normalize pixel values to be between 0 and 1 and resize images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    images = np.expand_dims(images, axis=-1)
    images = np.array([transform(image).numpy() for image in images])

    # Convert the labels to PyTorch tensors
    labels = torch.tensor(labels).long()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)


    # Create datasets
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Define a simple CNN model for binary classification using PyTorch
    class SimpleBinaryCNN(nn.Module):
        def __init__(self):
            super(SimpleBinaryCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
            self.fc1 = nn.Linear(64 * 32 * 32, 64)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, 1)  # Change to 1 output neuron

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 32 * 32)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Instantiate the model for binary classification
    model = SimpleBinaryCNN()

    # Define loss function for binary classification
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters())

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())  # Adjust for the BCEWithLogitsLoss
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output.squeeze(), target.float()).item()  # Update loss computation
                pred = output.round()  # Round to get binary predictions
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test accuracy: {test_accuracy:.4f}%")

    # Save the model
    torch.save(model.state_dict(), 'model2.pth')
