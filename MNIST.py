
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Transform images into tensors
transform = transforms.ToTensor()

# Load MNIST dataset
train_data = datasets.MNIST(
    root='./cnn_data',
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root='./cnn_data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(
    train_data,
    batch_size=10,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=10,
    shuffle=False
)

# Set random seed
seed = 41
torch.manual_seed(seed)


# Create the CNN
class ConvNN(nn.Module):
    def __init__(self, il, ol, ks, stl, ol2):
        super().__init__()

        self.conv1 = nn.Conv2d(il, ol, ks, stl)
        self.conv2 = nn.Conv2d(ol, ol2, ks, stl)

        # Image dimensions:
        # 28 -> 26 -> 13 -> 11 -> 5
        self.fc1 = nn.Linear(ol2 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)

        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)

        # Flatten feature maps
        x = x.view(x.size(0), -1)

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Initialize model
model = ConvNN(1, 6, 3, 1, 16)

# Loss function and optimizer
cri = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Training settings
Epochs = 10

train_losses = []
test_losses = []

train_correct = []
test_correct = []


# Train the model
for i in range(Epochs):

    model.train()

    trn_corr = 0
    train_loss = 0.0

    for b, (X_train, y_train) in enumerate(train_loader):

        # Make predictions
        y_pred = model(X_train)

        # Calculate loss
        loss = cri(y_pred, y_train)

        # Get predicted classes
        predicted = torch.max(y_pred, 1)[1]

        # Count correct predictions
        batch_corr = (predicted == y_train).sum().item()
        trn_corr += batch_corr

        # Update model weights
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()

        if (b + 1) % 600 == 0:
            print(
                f"Epoch: {i + 1}, "
                f"Batch: {b + 1}, "
                f"Loss: {loss.item():.4f}"
            )

    # Calculate training metrics
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * trn_corr / len(train_data)

    train_losses.append(avg_train_loss)
    train_correct.append(train_accuracy)

    # Evaluate model
    model.eval()

    tst_corr = 0
    test_loss = 0.0

    with torch.no_grad():
        for X_test, y_test in test_loader:

            y_val = model(X_test)

            loss = cri(y_val, y_test)

            predicted = torch.max(y_val, 1)[1]

            tst_corr += (predicted == y_test).sum().item()
            test_loss += loss.item()

    # Calculate testing metrics
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * tst_corr / len(test_data)

    test_losses.append(avg_test_loss)
    test_correct.append(test_accuracy)

    print(
        f"\nEpoch {i + 1}/{Epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"Test Accuracy: {test_accuracy:.2f}%\n"
    )


# Plot and save training/testing loss
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, Epochs + 1),
    train_losses,
    marker='o',
    label='Training Loss'
)

plt.plot(
    range(1, Epochs + 1),
    test_losses,
    marker='o',
    label='Testing Loss'
)

plt.title('Training and Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, Epochs + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('loss_graph.png', dpi=300, bbox_inches='tight')
plt.show()


# Plot and save training/testing accuracy
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, Epochs + 1),
    train_correct,
    marker='o',
    label='Training Accuracy'
)

plt.plot(
    range(1, Epochs + 1),
    test_correct,
    marker='o',
    label='Testing Accuracy'
)

plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(range(1, Epochs + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('accuracy_graph.png', dpi=300, bbox_inches='tight')
plt.show()


# Generate predictions for confusion matrix
all_predictions = []
all_labels = []

model.eval()

with torch.no_grad():
    for X_test, y_test in test_loader:

        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())


# Create confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

print("Confusion Matrix:")
print(cm)


# Display and save confusion matrix
plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('MNIST Confusion Matrix')
plt.colorbar()

classes = list(range(10))

plt.xticks(classes, classes)
plt.yticks(classes, classes)

plt.xlabel('Predicted Digit')
plt.ylabel('Actual Digit')

# Add values to every cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha='center',
            va='center',
            color='white' if cm[i, j] > cm.max() / 2 else 'black'
        )

plt.tight_layout()

plt.savefig(
    'confusion_matrix.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# Display 10 test images and their predictions
images, labels = next(iter(test_loader))

model.eval()

with torch.no_grad():
    predictions = model(images)
    predicted_labels = torch.max(predictions, 1)[1]


fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.flat):

    ax.imshow(images[i].squeeze(), cmap='gray')

    ax.set_title(
        f"True: {labels[i].item()}\n"
        f"Predicted: {predicted_labels[i].item()}"
    )

    ax.axis('off')

plt.tight_layout()

plt.savefig(
    'sample_predictions.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# Print saved image locations
print("\nAll graphs and images have been saved:")
print("1. loss_graph.png")
print("2. accuracy_graph.png")
print("3. confusion_matrix.png")
print("4. sample_predictions.png")
