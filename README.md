# Fruit Classification Models

A comprehensive implementation of three deep learning models for fruit classification using PyTorch: ConvNeXT, ResNet50, and Vision Transformer (ViT). Each model is specifically tuned for fruit image classification.

## Models Overview

### 1. ConvNeXT Classifier
- Implements the ConvNeXT architecture with hierarchical feature extraction
- Uses modern training techniques including AdamW optimizer and cosine learning rate scheduling
- Custom head architecture with LayerNorm and dropout for better generalization

### 2. ResNet50 Classifier
- Utilizes the ResNet50 architecture with pre-trained weights
- Features a custom classification head with additional fully connected layers
- Implements advanced data augmentation techniques
- Uses AdamW optimizer with weight decay for regularization

### 3. Vision Transformer (ViT) Classifier
- Implements the ViT-B/16 architecture with patch-based image processing
- Includes zero-shot classification capabilities using CLIP
- Features advanced augmentation with RandAugment
- Supports multi-GPU training with DataParallel

## Features

- Dataset management with automatic download and splitting
- Comprehensive data augmentation pipeline
- Training progress monitoring and visualization
- Model evaluation with detailed metrics
- GPU support with automatic device selection
- Zero-shot inference capabilities (ViT only)
- Best model checkpointing

## Requirements

```
torch
torchvision
transformers
PIL
sklearn
matplotlib
seaborn
kagglehub
numpy
```

## Dataset Structure

The code expects a dataset with the following structure:
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## Usage

### 1. Setup Dataset
```python
train_dir, val_dir, test_dir = setup_and_split_dataset()
```

### 2. Initialize Classifier
```python
# For ConvNeXT
classifier = ConvNextClassifier(
    num_classes=5,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001
)

# For ResNet50
classifier = ResNetClassifier(
    num_classes=5,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001
)

# For ViT
classifier = ViTClassifier(
    num_classes=5,
    batch_size=32,
    num_epochs=30,
    learning_rate=0.001
)
```

### 3. Train Model
```python
train_losses, val_accuracies = classifier.train_model(train_dir, val_dir)
```

### 4. Evaluate Model
```python
test_acc, test_metrics = classifier.evaluate(test_loader)
```

### 5. Visualize Results
```python
classifier.plot_results(train_losses, val_accuracies)
```

## Model Architecture Details

### ConvNeXT
- Hierarchical feature extraction
- Stage-wise processing with increasing channels
- Modern normalization and activation functions
- Optimized for contemporary vision tasks

### ResNet50
- Deep residual learning
- Skip connections for gradient flow
- Bottleneck blocks for efficiency
- Batch normalization for training stability

### Vision Transformer
- Patch-based image processing
- Self-attention mechanisms
- Position embeddings
- Optional zero-shot capabilities with CLIP

## Performance Monitoring

All models include:
- Training loss tracking
- Validation accuracy monitoring
- Precision, recall, and F1 score calculation
- Learning rate scheduling
- Model checkpointing

## Output and Visualization

The training process generates:
- Training loss plots
- Validation accuracy curves
- Detailed metrics for test set performance
- Model architecture summaries
- Training progress logs

## Notes

- All models support both CPU and GPU training
- Models automatically use multiple GPUs if available
- Dataset is automatically downloaded and split
- Best model weights are saved during training
- Comprehensive error handling and logging
- Configurable hyperparameters for all models

## License

This project is licensed under the MIT License - see the LICENSE file for details.