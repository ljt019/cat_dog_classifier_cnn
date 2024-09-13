import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Ensure that matplotlib does not try to open a window (useful if running on a server)
import matplotlib
matplotlib.use('Agg')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_config(config_file='config.json'):
    """
    Loads configuration parameters from a JSON file.
    Args:
        config_file (str): Path to the JSON config file.
    Returns:
        config (dict): Dictionary containing configuration parameters.
    """
    with open(config_file, 'r') as f:
        return json.load(f)

def download_quickdraw_data():
    """
    Downloads 'cat.npy' and 'dog.npy' files from the Quick, Draw! dataset.
    """
    os.makedirs('quickdraw_data', exist_ok=True)
    base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

    categories = ['cat', 'dog']
    for category in categories:
        url = f"{base_url}{category}.npy"
        save_path = os.path.join('quickdraw_data', f"{category}.npy")

        if os.path.exists(save_path):
            print(f"{category}.npy already exists, skipping download.")
            continue

        print(f"Downloading {category}.npy...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {category}.npy")
        else:
            print(f"Failed to download {category}.npy. Status code: {response.status_code}")

def load_and_preprocess_data(num_samples=5000):
    """
    Loads and preprocesses the data for 'cat' and 'dog' categories.
    Args:
        num_samples (int): Number of samples to load for each category.
    Returns:
        train_loader, test_loader: DataLoaders for training and testing.
    """
    # Load data
    cat_data = np.load('quickdraw_data/cat.npy')
    dog_data = np.load('quickdraw_data/dog.npy')

    # Limit the number of samples
    cat_data = cat_data[:num_samples]
    dog_data = dog_data[:num_samples]

    # Create labels: 0 for cat, 1 for dog
    cat_labels = np.zeros(len(cat_data), dtype=np.int64)
    dog_labels = np.ones(len(dog_data), dtype=np.int64)

    # Combine data and labels
    data = np.concatenate((cat_data, dog_data), axis=0)
    labels = np.concatenate((cat_labels, dog_labels), axis=0)

    # Normalize data
    data = data.astype('float32') / 255.0

    # Reshape data for PyTorch: (batch_size, channels, height, width)
    data = data.reshape(-1, 1, 28, 28)

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data)
    labels_tensor = torch.tensor(labels)

    # Create a TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)

    # Split dataset into training and testing sets (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    config = load_config()
    batch_size = config['batch_size']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

class SimpleCNN(nn.Module):
    """
    Defines a simple Convolutional Neural Network for binary classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 output classes: cat and dog

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Convolutional layer 1
        x = self.pool(x)           # Max pooling
        x = F.relu(self.conv2(x))  # Convolutional layer 2
        x = self.pool(x)           # Max pooling
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))    # Fully connected layer 1
        x = self.fc2(x)            # Output layer
        return x

def train_model(model, train_loader, num_epochs=5, learning_rate=0.001):
    """
    Trains the model using the training DataLoader.
    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

def evaluate_model(model, test_loader):
    """
    Evaluates the model on the test DataLoader.
    Args:
        model: The trained neural network model.
        test_loader: DataLoader for the test data.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def save_model(model, filepath='cat_dog_classifier.pth'):
    """
    Saves the trained model to a file.
    Args:
        model: The trained neural network model.
        filepath (str): The path where the model will be saved.
    """
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')

def load_model(model, filepath='cat_dog_classifier.pth'):
    """
    Loads the model parameters from a file.
    Args:
        model: The neural network model to load parameters into.
        filepath (str): The path to the saved model file.
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f'Model loaded from {filepath}')

def predict_image(model, image):
    """
    Predicts the class of a single image.
    Args:
        model: The trained neural network model.
        image: A PIL Image or NumPy array.
    Returns:
        prediction (str): The predicted class label ('cat' or 'dog').
    """
    # Preprocess the image
    if isinstance(image, Image.Image):
        image = image.resize((28, 28)).convert('L')
        image = np.array(image).astype('float32') / 255.0
    elif isinstance(image, np.ndarray):
        if image.shape != (28, 28):
            image = Image.fromarray(image).resize((28, 28)).convert('L')
            image = np.array(image).astype('float32') / 255.0
    else:
        raise ValueError("Image must be a PIL Image or NumPy array.")

    image = image.reshape(1, 1, 28, 28)
    image_tensor = torch.tensor(image).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    return 'cat' if predicted.item() == 0 else 'dog'

def visualize_predictions(model, test_loader, num_images=8):
    """
    Visualizes sample predictions from the test set.
    Args:
        model: The trained neural network model.
        test_loader: DataLoader for the test data.
        num_images (int): Number of images to display.
    """
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)  # Use the built-in next() function

    images = images.to(device)
    labels = labels.to(device)

    # Get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Move images to CPU for plotting
    images = images.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    # Plot the images with predicted and true labels
    fig = plt.figure(figsize=(10, 4))
    for idx in range(num_images):
        ax = fig.add_subplot(2, num_images // 2, idx+1)
        img = images[idx][0]
        ax.imshow(img, cmap='gray')
        pred_label = 'cat' if predicted[idx] == 0 else 'dog'
        true_label = 'cat' if labels[idx] == 0 else 'dog'
        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    print('Sample predictions saved to sample_predictions.png')

def main():
    # Load configuration
    config = load_config()

    # Step 1: Download the data
    download_quickdraw_data()

    # Step 2: Load and preprocess the data
    train_loader, test_loader = load_and_preprocess_data(num_samples=config['num_samples'])

    # Step 3: Initialize the model
    model = SimpleCNN().to(device)

    # Step 4: Train the model
    train_model(model, train_loader, num_epochs=config['num_epochs'], learning_rate=config['learning_rate'])

    # Step 5: Evaluate the model
    evaluate_model(model, test_loader)

    # Step 6: Visualize sample predictions
    visualize_predictions(model, test_loader, num_images=8)

    # Step 7: Save the model
    save_model(model, config['model_save_path'])

if __name__ == '__main__':
    main()
