
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix


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