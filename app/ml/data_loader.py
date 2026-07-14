import os
import glob
from typing import List, Tuple, Dict

def get_image_paths_and_labels(data_dir: str) -> Tuple[List[str], List[int], Dict[int, str]]:
    """
    Scans the data directory and extracts image file paths and their corresponding labels.
    Assumes a directory structure where subdirectories are class names.
    e.g., data_dir/def_front/img1.jpeg, data_dir/ok_front/img2.jpeg
    
    Args:
        data_dir: Path to the raw data directory containing class subdirectories.
        
    Returns:
        A tuple containing:
        - List of image file paths
        - List of integer labels
        - Dictionary mapping integer labels to class names
    """
    image_paths = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist.")
        return [], [], {}
        
    # Identify classes based on subdirectories
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        # The casting dataset images are typically jpeg
        for ext in ('*.jpeg', '*.jpg', '*.png'):
            for img_path in glob.glob(os.path.join(cls_dir, ext)):
                image_paths.append(img_path)
                labels.append(class_to_idx[cls_name])
                
    return image_paths, labels, idx_to_class
