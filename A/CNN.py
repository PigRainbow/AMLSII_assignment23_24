#Import required libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Data preparation
class imagedataset(Dataset):
    """corn dataset"""

    # Read data and preprocess data
    def __init__(self, csv_file, image_dir, transform=None):

        self.image_labels = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    # Return the size of the dataset
    def __len__(self):
        return len(self.image_labels)

    # Return one sample at a time
    def __getitem__(self, idx):
        image_file = str(self.image_labels.iloc[idx, 2])
        if image_file.startswith('train/'):
          image_file = image_file[len('train/'):]
        elif image_file.startswith('test/'):
          image_file = image_file[len('test/'):]
        image_path = os.path.join(self.image_dir, image_file)
        image = read_image(image_path)
        self.label_to_idx = {'pure': 0, 'broken': 1, 'discolored': 2, 'silkcut': 3}
        label_str = self.image_labels.iloc[idx, 3]
        label = self.label_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


# Access to the dataset file
train_csv_file_path =  '../Datasets/train.csv'
test_csv_file_path = '../Datasets/test.csv'
train_image_path = '../Datasets/train'
test_image_path = '../Datasets/test'

# Data augmentation
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(degrees=45), 
    transforms.ColorJitter(),  
    transforms.ConvertImageDtype(torch.float),

])

test_tfm = transforms.Compose([
    # Resize the image into a fixed shape
    transforms.Resize((128, 128)),
    transforms.ConvertImageDtype(torch.float),
])

# Instantiate training set, validation set and testing set
train_dataset = imagedataset(
    csv_file=train_csv_file_path,
    image_dir=train_image_path,
    transform=train_tfm
)

test_dataset = imagedataset(
    csv_file=test_csv_file_path,
    image_dir=test_image_path,
    transform=test_tfm
)

# Split original training dataset 90% for training set, 10% for validation set
original_train_set = len(train_dataset)
split_ratio = 0.1
val_size = int(original_train_set * split_ratio)
train_size = original_train_set - val_size
train_new_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


# Creat DataLoader
batch_value = 128
train_loader = DataLoader(train_new_dataset, batch_size=batch_value, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_value, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_value, shuffle=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


# Training and validation in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set list to collect accuracy and loss
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

# Train
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()  # Set loss function for classification task
optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # Other learning rate: 0.01, 0.005
#optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
n_epochs = 20  # Number of training epochs, other epoch setting: 35, 50, 80

for epoch in range(n_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) # Move data to device
        optimizer.zero_grad()  # Clear the gradients of all optimized tensors
        outputs = model(inputs) # Forward pass: compute predicted outputs by passing inputs to the model
        loss = criterion(outputs, labels) # Compute the loss
        loss.backward() # Backward pass: compute gradient of the loss with respect to model parameters(backpropagation)
        optimizer.step()  # updtae model with optimizer

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print(f'Training: Epoch {epoch+1}/{n_epochs} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')

    # Validation
    # Set model to evaluation model
    model.eval()
    val_total_loss = 0
    val_total_corrects = 0

    # Iterate through the dataloader
    for inputs, labels in val_loader:
      inputs, labels = inputs.to(device), labels.to(device) # Move data to device

      # Disable gradient calculation
      with torch.no_grad():
        outputs = model(inputs) # Forward pass: compute predicted outputs by passing inputs to the model

        # Compute the loss
        val_loss = criterion(outputs, labels)
        val_total_loss += val_loss.item() * inputs.size(0)

        # Compute the accuracy
        _, preds = torch.max(outputs, 1)
        val_total_corrects += torch.sum(preds == labels.data)

    # Compute the average loss and average accuracy
    val_avg_loss = val_total_loss / len(val_loader.dataset)
    val_avg_accuracy = val_total_corrects.double() / len(val_loader.dataset)

    print(f'Validation Loss after Epoch {epoch+1}: {val_avg_loss:.4f}, Validation Accuracy: {val_avg_accuracy:.4f}')

    # After each epoch, accuracy and loss data are collected and transformed
    train_accuracies.append(running_corrects.double().cpu().numpy() / len(train_loader.dataset))
    val_accuracies.append(val_total_corrects.double().cpu().numpy() / len(val_loader.dataset))
    train_losses.append(epoch_loss)
    val_losses.append(val_avg_loss)


# After training, plot the accuracies
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, n_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Training and Validation Accuracy')
plt.xticks(range(0, n_epochs + 1, 5))
plt.legend()
plt.show()

# Plotting losses
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xticks(range(0, n_epochs + 1, 5))
plt.show()

# Set model to evaluation mode
model.eval()

# Create lists to store predictions and true labels
all_preds = []
all_labels = []

# Iterate through the dataloader
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Disable gradient calculation
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to NumPy arrays for accuracy calculation
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate overall accuracy
overall_accuracy = accuracy_score(all_labels, all_preds)

# Calculate per-class accuracy and other metrics
class_report = classification_report(all_labels, all_preds, target_names=['pure', 'broken', 'discolored', 'silkcut'])

print(f'Overall Accuracy: {overall_accuracy:.4f}')
print(f'Per Class Accuracy and other metrics:\n{class_report}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=['pure', 'broken', 'discolored', 'silkcut'], yticklabels=['pure', 'broken', 'discolored', 'silkcut'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
