import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ViT_B_16_Weights
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
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import numpy as np

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

class ViTClassifier:
    def __init__(self, num_classes=5, batch_size=32, num_epochs=30, learning_rate=0.001):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        torch.backends.cudnn.benchmark = True  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        else:
            print("WARNING: GPU not available, using CPU")
        
        self.architecture_config = {
            'image_size': 224,
            'patch_size': 16,
            'num_channels': 3,
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 12,
            'mlp_dim': 3072,
            'dropout': 0.1,
            'attention_dropout': 0.0,
            'num_classes': num_classes
        }
        
        self._setup_transforms()
        
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.05  
        )

    def calculate_class_similarity(image_features, text_features, num_descriptions_per_class=5):
        """Calculate similarity scores aggregating multiple descriptions per class"""
        similarity = (100.0 * image_features @ text_features.T)
        
        batch_size = similarity.shape[0]
        num_classes = similarity.shape[1] // num_descriptions_per_class
        
        similarity = similarity.view(batch_size, num_classes, num_descriptions_per_class)
        
        class_similarity = torch.max(similarity, dim=2)[0]
        
        return torch.softmax(class_similarity, dim=-1)

    def zero_shot_inference(self, test_loader, class_descriptions):
        """Perform zero-shot inference using CLIP-like text-image matching"""
        print("Loading CLIP model for zero-shot inference...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        clip_model = clip_model.to(self.device)
        clip_model.eval()
        
        flat_descriptions = [desc for class_descs in class_descriptions for desc in class_descs]
        text_inputs = processor(
            text=flat_descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        text_features = clip_model.get_text_features(**{k: v.to(self.device) for k, v in text_inputs.items()})
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        all_preds = []
        all_labels = []
        
        print("Performing zero-shot inference...")
        with torch.no_grad():
            for images, labels in test_loader:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                images = images.to(self.device)
                denormalized_images = images * std + mean
                
                images_pil = []
                for img in denormalized_images:
                    img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                    images_pil.append(Image.fromarray(img_np))
                
                image_inputs = processor(
                    images=images_pil,
                    return_tensors="pt"
                )
                
                image_features = clip_model.get_image_features(
                    **{k: v.to(self.device) for k, v in image_inputs.items()}
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T)
                
                num_descriptions_per_class = len(class_descriptions[0]) 
                similarity = similarity.view(similarity.shape[0], -1, num_descriptions_per_class)
                class_similarity = similarity.mean(dim=-1)  
                
                predictions = class_similarity.argmax(dim=-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        metrics = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        return accuracy, {
            'precision': metrics[0],
            'recall': metrics[1],
            'f1': metrics[2]
        }

    def _setup_transforms(self):
        """Setup data transformations for ViT"""
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandAugment(num_ops=2, magnitude=9),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_model(self):
        """Create and configure the ViT model"""
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        print("\nViT-B/16 Architecture Details:")
        print("==============================")
        print("Patch Embedding:")
        print(f"- Image size: {self.architecture_config['image_size']}x{self.architecture_config['image_size']}")
        print(f"- Patch size: {self.architecture_config['patch_size']}x{self.architecture_config['patch_size']}")
        print(f"- Number of patches: {(self.architecture_config['image_size']//self.architecture_config['patch_size'])**2}")
        print(f"- Embedding dimension: {self.architecture_config['embed_dim']}")
        
        print("\nTransformer Encoder:")
        print(f"- Number of layers: {self.architecture_config['num_layers']}")
        print(f"- Number of heads: {self.architecture_config['num_heads']}")
        print(f"- MLP dimension: {self.architecture_config['mlp_dim']}")
        print(f"- Dropout: {self.architecture_config['dropout']}")
        
        num_features = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_classes)
        )
        
        return model

    def train_model(self, train_dir, val_dir, class_descriptions=None):
        """Train the ViT model with optional zero-shot evaluation first"""
        print("Creating datasets...")
        train_dataset = FruitDataset(train_dir, transform=self.train_transform)
        val_dataset = FruitDataset(val_dir, transform=self.val_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,  
            pin_memory=True  
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        if class_descriptions:
            print("\nPerforming zero-shot evaluation before fine-tuning...")
            zero_shot_acc, zero_shot_metrics = self.zero_shot_inference(val_loader, class_descriptions)
            print(f"Zero-shot Accuracy: {zero_shot_acc:.4f}")
            print(f"Zero-shot Metrics: {zero_shot_metrics}")
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
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
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
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
                torch.save(self.model.state_dict(), 'best_vit_model.pth')
                print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
        return train_losses, val_accuracies

    def evaluate(self, dataloader):
        """Evaluate the ViT model"""
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
        plt.savefig('vit_training_results.png')
        plt.show()

def test_pretrained_vit(test_dir):
    """Test pre-trained ViT-B/16 before any fine-tuning"""
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    imagenet_fruit_indices = {
        'Apple': [948, 949, 950],  
        'Banana': [954],           
        'Grape': [951, 952],      
        'Mango': [956],            
        'Strawberry': [949]        
    }
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FruitDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print("Testing pre-trained ViT-B/16 model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            
            batch_preds = []
            for prob in probs:
                class_scores = []
                for fruit_class in test_dataset.classes:
                    class_score = sum(prob[idx].item() for idx in imagenet_fruit_indices[fruit_class])
                    class_scores.append(class_score)
                pred = np.argmax(class_scores)
                batch_preds.append(pred)
            
            all_preds.extend(batch_preds)
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print("\nPre-trained ViT-B/16 Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    print("Setting up dataset...")
    random.seed(42)
    train_dir, val_dir, test_dir = setup_and_split_dataset()

    print("\nTesting pre-trained ViT-B/16...")
    pretrained_acc, pretrained_metrics = test_pretrained_vit(test_dir)
    
    class_descriptions = [
    [
        "a photo of a fresh red apple",
        "a close-up photo of a shiny apple fruit",
        "a photograph of a whole apple with stem and leaves",
        "an image of a round red or green apple",
        "a detailed shot of an apple's smooth skin"
    ],
    [
        "a photo of yellow ripe bananas",
        "a bunch of curved yellow bananas",
        "a photograph of fresh banana fruit with peel",
        "an image of bright yellow banana cluster",
        "a close-up of banana's distinctive curved shape"
    ],
    [
        "a photo of a round orange citrus fruit",
        "a close-up of an orange with textured peel",
        "a bright orange citrus fruit with pitted skin",
        "an image of a whole fresh orange",
        "a detailed shot of orange fruit with dimpled surface"
    ],
    [
        "a photo of a fresh green pear",
        "an image of a pear's distinctive teardrop shape",
        "a photograph of a ripe pear with stem",
        "a close-up of a pear's smooth skin",
        "a detailed shot of a yellow-green pear fruit"
    ],
    [
        "a photo of an oval-shaped mango fruit",
        "an image of a ripe yellow-red mango",
        "a photograph of a fresh tropical mango",
        "a close-up of mango's smooth skin",
        "a detailed shot of mango's distinctive oblong shape"
    ]
]
    flat_descriptions = [desc for class_descs in class_descriptions for desc in class_descs]

    num_classes = 5
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    
    print("Initializing classifier...")
    classifier = ViTClassifier(
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    print("Starting training and evaluation...")
    train_losses, val_accuracies = classifier.train_model(
        train_dir, 
        val_dir,
        class_descriptions=class_descriptions  
    )
    
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
    
    print("\nModel Architecture Summary:")
    print("==========================")
    print("Vision Transformer (ViT) Characteristics:")
    print("- Self-attention based architecture")
    print("- Global receptive field from the start")
    print("- Patch-based image processing")
    print("- Position embeddings for spatial information")
    print("- Advanced augmentation with RandAugment")
    
    return classifier, test_acc, test_metrics

if __name__ == "__main__":
    main()