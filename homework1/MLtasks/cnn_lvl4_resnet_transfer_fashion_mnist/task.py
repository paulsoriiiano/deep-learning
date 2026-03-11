"""
Transfer Learning with ResNet on Fashion-MNIST
Fine-tuning pretrained ResNet18 for Fashion-MNIST classification
"""

import os
import json
import copy
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Device (CUDA -> MPS -> CPU)
# -----------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

# Output directory (adjust if your task runner expects a specific path)
# OUTPUT_DIR = "/Developer/AIserver/output/tasks/cnn_lvl4_resnet_transfer_fashion_mnist"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the ML task."""
    return {
        "task_type": "image_classification",
        "dataset": "FashionMNIST",
        "num_classes": 10,
        "model_type": "ResNet18",
        "transfer_learning": True,
        # We convert grayscale -> 3ch and resize -> 224 for pretrained ResNet
        "input_shape": [3, 224, 224],
        "description": "Fine-tuning pretrained ResNet18 on Fashion-MNIST dataset",
    }


def make_dataloaders(batch_size=64, val_ratio=0.2, num_workers=None):
    """
    Create data loaders for Fashion-MNIST dataset.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # MPS environments are often more stable with num_workers=0
    if num_workers is None:
        num_workers = 0 if device.type == "mps" else 2

    # pin_memory helps CUDA transfers; avoid for MPS/CPU
    pin = (device.type == "cuda")

    # ImageNet normalization (recommended when using ImageNet-pretrained weights)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225],   # ImageNet std
    )

    # Fashion-MNIST is 1x28x28. Convert to 3ch and resize to 224.
    # Light augmentation for training
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    full_train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=train_transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=eval_transform
    )

    # Split train into train/val
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))

    np.random.shuffle(indices)
    train_indices = indices[split:]
    val_indices = indices[:split]

    train_subset = Subset(full_train_dataset, train_indices)

    # IMPORTANT: val subset should NOT use training augmentation.
    full_train_eval = datasets.FashionMNIST(
        root="./data", train=True, download=False, transform=eval_transform
    )
    val_subset = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes=10, pretrained=True, freeze_base=True):
    """
    Build and configure ResNet model for Fashion-MNIST.
    """
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)

    # Replace classifier head
    model.fc = nn.Linear(512, num_classes)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    return model.to(device)


def convert_to_python_scalars(obj):
    """Recursively convert tensors and numpy arrays to Python scalars for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_python_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_python_scalars(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def evaluate(model, data_loader, criterion):
    """Evaluate the model on given data loader."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(data_loader.dataset)
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_targets, all_predictions)

    class_report = classification_report(
        all_targets, all_predictions, output_dict=True, zero_division=0
    )

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "class_report": class_report,
        "num_samples": total_samples,
    }


def train(model, train_loader, val_loader, criterion, optimizer,
          num_epochs=10, scheduler=None, early_stopping_patience=5):
    """Train the model with optional early stopping."""
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0,
        "best_model_state": None,
    }

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total

        val_results = evaluate(model, val_loader, criterion)

        history["train_loss"].append(float(train_epoch_loss))
        history["train_acc"].append(float(train_epoch_acc))
        history["val_loss"].append(float(val_results["loss"]))
        history["val_acc"].append(float(val_results["accuracy"]))

        if scheduler is not None:
            scheduler.step(val_results["loss"])

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f} | "
            f"Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']:.4f}"
        )

        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            history["best_val_acc"] = float(best_val_acc)
            history["best_model_state"] = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if history["best_model_state"] is not None:
        model.load_state_dict(history["best_model_state"])

    return model, history


def save_artifacts(model, history, class_names, output_dir='./output', test_results=None):
    """Save model artifacts and evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    clean_history = {}
    for k, v in history.items():
        if k == "best_model_state":
            continue
        clean_history[k] = convert_to_python_scalars(v)

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(clean_history, f, indent=2)

    class_names_path = os.path.join(output_dir, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)

    if test_results is not None:
        test_results_path = os.path.join(output_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_results, f, indent=2)

    print(f"Artifacts saved to {output_dir}")


def main():
    print("=" * 60)
    print("Transfer Learning with ResNet on Fashion-MNIST")
    print("=" * 60)

    metadata = get_task_metadata()
    print(f"\nTask: {metadata['description']}")
    print(f"Dataset: {metadata['dataset']} ({metadata['num_classes']} classes)")
    print(f"Device: {device} (type={device.type})")

    print("\n[1/5] Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = make_dataloaders(
        batch_size=64, val_ratio=0.2, num_workers=None
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print("\n[2/5] Building model (frozen base)...")
    model = build_model(
        num_classes=metadata["num_classes"], pretrained=True, freeze_base=True
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    print("\n[3/5] Phase 1: Training final layer only...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    model, history = train(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=8, scheduler=scheduler, early_stopping_patience=4
    )

    print("\n[4/5] Phase 2: Unfreezing and fine-tuning...")
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    model, ft_history = train(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=10, scheduler=scheduler, early_stopping_patience=4
    )

    # Merge histories
    for k in ["train_loss", "train_acc", "val_loss", "val_acc"]:
        history[k].extend(ft_history[k])
    history["best_val_acc"] = float(max(history["val_acc"])) if history["val_acc"] else 0.0

    print("\n[5/5] Evaluating model...")
    train_results = evaluate(model, train_loader, criterion)
    val_results = evaluate(model, val_loader, criterion)
    test_results = evaluate(model, test_loader, criterion)

    print("\nTrain Results:")
    print(f" Loss: {train_results['loss']:.4f}")
    print(f" Accuracy: {train_results['accuracy']:.4f}")

    print("\nValidation Results:")
    print(f" Loss: {val_results['loss']:.4f}")
    print(f" Accuracy: {val_results['accuracy']:.4f}")

    print("\nTest Results:")
    print(f" Loss: {test_results['loss']:.4f}")
    print(f" Accuracy: {test_results['accuracy']:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model, history, class_names, test_results)

    print("\n" + "=" * 60)
    print("Task Complete")
    print("=" * 60)


    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy:  {train_results['accuracy']:.4f}")
    print(f"Val Accuracy:    {val_results['accuracy']:.4f}")
    print(f"Train Loss:      {train_results['loss']:.4f}")
    print(f"Val Loss:        {val_results['loss']:.4f}")

    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    # Check 1: Train accuracy > 0.92
    check1 = train_results['accuracy'] > 0.92
    status1 = "✓" if check1 else "✗"
    print(f"{status1} Train Accuracy > 0.92: {train_results['accuracy']:.4f}")
    checks_passed = checks_passed and check1

    # Check 2: Val accuracy > 0.92
    check2 = val_results['accuracy'] > 0.92
    status2 = "✓" if check2 else "✗"
    print(f"{status2} Val Accuracy > 0.92: {val_results['accuracy']:.4f}")
    checks_passed = checks_passed and check2

    # Check 3: Val loss < 0.30
    check3 = val_results['loss'] < 0.30
    status3 = "✓" if check3 else "✗"
    print(f"{status3} Val Loss < 0.30: {val_results['loss']:.4f}")
    checks_passed = checks_passed and check3

    # Check 4: No significant overfitting (accuracy gap < 0.05)
    accuracy_gap = abs(train_results['accuracy'] - val_results['accuracy'])
    check4 = accuracy_gap < 0.05
    status4 = "✓" if check4 else "✗"
    print(f"{status4} Accuracy gap < 0.05: {accuracy_gap:.4f}")
    checks_passed = checks_passed and check4

    # Final summary
    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    main()