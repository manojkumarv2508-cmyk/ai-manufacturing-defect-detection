import torch
import torch.nn as nn
import torchvision.models as models

def get_defect_detection_model(num_classes: int = 2) -> nn.Module:
    """
    Initializes a ResNet18 model for defect detection using transfer learning.
    
    Args:
        num_classes: The number of output classes.
        
    Returns:
        A PyTorch model with a modified final layer.
    """
    # Load pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
        
    # Replace the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
