
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from timm import create_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

dataset_dir = "/content/drive/MyDrive/Disease Grading"


# Dataset definition
class IDRiDDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        self.images_dir = images_dir
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
      img_name = self.labels.iloc[idx, 0]
      possible_extensions = ['.png', '.jpg', '.jpeg']
      img_path = None

      for ext in possible_extensions:
          potential_path = os.path.join(self.images_dir, img_name + ext)
          if os.path.exists(potential_path):
              img_path = potential_path
              break

      if img_path is None:
          raise FileNotFoundError(f"File {img_name} with supported extensions not found.")

      image = Image.open(img_path).convert("RGB")
      label = self.labels.iloc[idx, 1]
      if self.transform:
          image = self.transform(image)
      return image, label


# Paths
dataset_dir = "/content/drive/MyDrive/Disease Grading"
train_images_dir = os.path.join(dataset_dir, "Images", "Train")
test_images_dir = os.path.join(dataset_dir, "Images", "Test")
train_labels_csv = os.path.join(dataset_dir, "GroundTruths", "train_labels.csv")
test_labels_csv = os.path.join(dataset_dir, "GroundTruths", "test_labels.csv")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit ViT input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datasets and DataLoaders
train_dataset = IDRiDDataset(train_images_dir, train_labels_csv, transform=transform)
test_dataset = IDRiDDataset(test_images_dir, test_labels_csv, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ViT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model("vit_base_patch16_224", pretrained=True, num_classes=5)  # Assuming 5 disease classes
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training and Evaluation
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

# Training Loop
num_epochs = 20
best_acc = 0.0
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # Save the best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_vit_model.pth")
        print(f"Saved Best Model with Test Acc: {best_acc:.2f}%")

# Plot Learning Curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.show()
