import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms.functional as TF
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import shutil
import random

def setup_and_split_dataset():
    """Download and split dataset into train/val/test"""
    try:
        dataset_path = kagglehub.dataset_download("utkarshsaxenadn/fruits-classification")
        print(f"Dataset downloaded to: {dataset_path}")
        
        base_path = Path(dataset_path) / "Fruits Classification"
        source_train_dir = base_path / "train"
        source_valid_dir = base_path / "valid"
        
        output_base = base_path / "split_dataset"
        train_dir = output_base / "train"
        val_dir = output_base / "val"
        test_dir = output_base / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("Splitting dataset into train/val/test...")
        
        for class_name in os.listdir(source_train_dir):
            print(f"Processing class: {class_name}")
            
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
            (test_dir / class_name).mkdir(exist_ok=True)
            
            all_images = []
            
            train_class_dir = source_train_dir / class_name
            if train_class_dir.exists():
                all_images.extend(list(train_class_dir.glob('*.jpg')))
                all_images.extend(list(train_class_dir.glob('*.jpeg')))
            
            valid_class_dir = source_valid_dir / class_name
            if valid_class_dir.exists():
                all_images.extend(list(valid_class_dir.glob('*.jpg')))
                all_images.extend(list(valid_class_dir.glob('*.jpeg')))
            
            random.shuffle(all_images)
            
            total_images = len(all_images)
            train_size = int(0.8 * total_images)
            val_size = int(0.1 * total_images)
            
            train_images = all_images[:train_size]
            val_images = all_images[train_size:train_size + val_size]
            test_images = all_images[train_size + val_size:]
            
            for img_path in train_images:
                shutil.copy2(img_path, train_dir / class_name / img_path.name)
            
            for img_path in val_images:
                shutil.copy2(img_path, val_dir / class_name / img_path.name)
            
            for img_path in test_images:
                shutil.copy2(img_path, test_dir / class_name / img_path.name)
            
            print(f"  Split sizes for {class_name}:")
            print(f"    Train: {len(train_images)}")
            print(f"    Val: {len(val_images)}")
            print(f"    Test: {len(test_images)}")
        
        return train_dir, val_dir, test_dir
        
    except Exception as e:
        print(f"Error setting up dataset: {str(e)}")
        raise

class FruitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name

            for extension in ['*.jpg', '*.jpeg']:
                for img_path in class_dir.glob(extension):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
                
        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ResNetClassifier:
    def __init__(self, num_classes=5, batch_size=32, num_epochs=10, learning_rate=0.001):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.architecture_config = {
            'layers': [3, 4, 6, 3],  
            'channels': [64, 128, 256, 512],  
            'expansion': 4, 
            'initial_kernel': 7,
            'initial_stride': 2,
            'initial_channels': 64
        }
        
        self._setup_transforms()
        
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

    def _setup_transforms(self):
        """Setup data transformations for ResNet"""
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _create_model(self):
        """Create and configure the ResNet50 model"""
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        print("\nResNet50 Architecture Details:")
        print("==============================")
        print("Initial Layer:")
        print(f"- Kernel size: {self.architecture_config['initial_kernel']}x{self.architecture_config['initial_kernel']}")
        print(f"- Initial stride: {self.architecture_config['initial_stride']}")
        print(f"- Initial channels: {self.architecture_config['initial_channels']}")
        
        print("\nStage Configuration:")
        total_params = 0
        for i, (layers, channels) in enumerate(zip(self.architecture_config['layers'], 
                                                 self.architecture_config['channels']), 1):
            channels_out = channels * self.architecture_config['expansion']
            print(f"\nStage {i}:")
            print(f"- Number of blocks: {layers}")
            print(f"- Input channels: {channels}")
            print(f"- Output channels: {channels_out}")
            
            params_per_block = (
                channels * channels_out +
                9 * channels * channels +
                channels * channels_out +
                (channels * channels_out if i > 1 else 0)
            )
            stage_params = params_per_block * layers
            total_params += stage_params
            print(f"- Approximate parameters: {stage_params:,}")
        
        print(f"\nTotal approximate parameters: {total_params:,}")
        
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return model

    def train_model(self, train_dir, val_dir):
        """Train the ResNet model"""
        print("Creating datasets...")
        train_dataset = FruitDataset(train_dir, transform=self.train_transform)
        val_dataset = FruitDataset(val_dir, transform=self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        print("Starting training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                running_loss += loss.item()
                
                if i % 20 == 0:
                    print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            val_acc, val_metrics = self.evaluate(val_loader)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            print(f'Training Loss: {epoch_loss:.4f}')
            print(f'Validation Accuracy: {val_acc:.4f}')
            print(f'Validation Metrics: {val_metrics}\n')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_resnet_model.pth')
                print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
        return train_losses, val_accuracies

    def evaluate(self, dataloader):
        """Evaluate the ResNet model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return accuracy, metrics

    def plot_results(self, train_losses, val_accuracies):
        """Plot training results"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()

def main():
    print("Setting up dataset...")
    random.seed(42)
    train_dir, val_dir, test_dir = setup_and_split_dataset()
    
    num_classes = 5
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    print("Initializing classifier...")
    classifier = ResNetClassifier(
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    print("Starting training and evaluation...")
    train_losses, val_accuracies = classifier.train_model(train_dir, val_dir)
    
    print("\nEvaluating on test set...")
    test_dataset = FruitDataset(test_dir, transform=classifier.val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_acc, test_metrics = classifier.evaluate(test_loader)
        
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
        
    classifier.plot_results(train_losses, val_accuracies)

if __name__ == "__main__":
        main()
