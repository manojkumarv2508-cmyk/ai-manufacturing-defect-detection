import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
    """
    Loads an image from the given path, resizes it, and normalizes pixel values.
    
    Args:
        image_path: Path to the image file.
        target_size: A tuple (width, height) to resize the image to.
        
    Returns:
        A numpy array of the preprocessed image with normalized values (0 to 1).
    """
    # Using IMREAD_GRAYSCALE as the industrial casting images are typically grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
        
    # Resize image to the target uniform size
    img_resized = cv2.resize(img, target_size)
    
    # Normalize pixel values to range [0, 1] for neural network stability
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Expand dimensions to have a channel (H, W, 1) required for many CNN architectures
    img_normalized = np.expand_dims(img_normalized, axis=-1)
    
    return img_normalized

def split_dataset(X: List[Any], y: List[int], test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    """
    Splits the dataset into training, validation, and test sets, maintaining class distribution.
    
    Args:
        X: List of image paths (or features).
        y: List of labels.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the TOTAL dataset to use for validation.
        random_state: Seed for reproducibility.
        
    Returns:
        Tuple containing splits: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First, separate out the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Calculate validation size relative to the remaining training data
    # If val_size is 0.15 of total, and test_size is 0.15, then train_val is 0.85
    # So relative val_size is 0.15 / 0.85
    relative_val_size = val_size / (1.0 - test_size)
    
    # Split the remaining data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, 
        random_state=random_state, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
