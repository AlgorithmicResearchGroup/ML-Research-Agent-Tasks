import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import timm
from collections import Counter

# Load the dataset
dataset = load_dataset("AlgorithmicResearchGroup/budget_model_train")

# Print dataset info
print(f"Train set size: {len(dataset['train'])}")
print(f"Validation set size: {len(dataset['val'])}")
print(f"Number of classes: {len(set(dataset['train']['class']))}")

# Check class balance
class_counts = Counter(dataset['train']['class'])
print("Class distribution:")
print(class_counts.most_common(5))  # Print the 5 most common classes
print(class_counts.most_common()[-5:])  # Print the 5 least common classes

# Define transforms with additional augmentations
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create PyTorch datasets
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, item['class']

train_dataset = ImageDataset(dataset['train'], transform_train)
val_dataset = ImageDataset(dataset['val'], transform_val)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# Set up the model
model = timm.create_model('resnet50', pretrained=True, num_classes=len(set(dataset['train']['class'])))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

# Learning rate scheduler with warmup
def lr_lambda(epoch):
    if epoch < 5:
        return (epoch + 1) / 5  # Warm up for 5 epochs
    else:
        return 1.0  # Constant LR after warmup

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    
    # Step the scheduler
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], LR: {scheduler.get_last_lr()[0]:.6f}, '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    # Early stopping (optional)
    if val_accuracy > 95:  # You can adjust this threshold
        print(f"Reached {val_accuracy:.2f}% validation accuracy. Stopping training.")
        break

# Save the trained model
torch.save(model.state_dict(), 'resnet50_classifier.pth')

print("Training completed. Model saved as 'resnet50_classifier.pth'")