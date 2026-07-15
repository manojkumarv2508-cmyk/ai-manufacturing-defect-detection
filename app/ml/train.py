import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    save_dir: str = "models",
    model_name: str = "defect_detector_resnet18.pth",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains the PyTorch model and evaluates it on the validation set.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ----------------------------
        # Training
        # ----------------------------
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()

        running_val_loss = 0.0
        running_corrects = 0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                running_val_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double().item() / len(val_loader.dataset)

        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(best_model_wts, save_path)

            print(f"Saved best model to {save_path}")

    model.load_state_dict(best_model_wts)

    return model, history