import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torchvision.transforms as transforms



# Load the test dataset
test_dataset = ImageFolder('path/to/test/dataset', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the saved model
model = torch.load('path/to/saved/model.pth')
model.eval()

# Define the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function (if needed)
criterion = torch.nn.CrossEntropyLoss()

# Define variables to keep track of accuracy and loss
total_correct = 0
total_loss = 0

# Iterate over the test dataset
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    
    # Calculate loss (if needed)
    loss = criterion(outputs, labels)
    total_loss += loss.item()

    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    total_correct += (predicted == labels).sum().item()

# Calculate average loss and accuracy
average_loss = total_loss / len(test_dataset)
accuracy = total_correct / len(test_dataset)

print(f'Average Loss: {average_loss:.4f}')
print(f'Accuracy: {accuracy * 100:.2f}%')