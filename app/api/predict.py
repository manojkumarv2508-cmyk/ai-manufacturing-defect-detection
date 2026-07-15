import time
import os
import io
import torch
from PIL import Image
import torchvision.transforms as transforms

from app.ml.model import get_defect_detection_model
from app.api.schemas import PredictionResponse

# Setup device to use CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for model state
MODEL = None
# Class names mapping, aligning with the outputs from Milestone 1 (0: def_front, 1: ok_front)
CLASS_NAMES = ["def_front", "ok_front"]

def load_model():
    """
    Loads the trained model weights into memory. 
    This should be called only once on startup.
    """
    global MODEL
    if MODEL is None:
        print(f"Loading model on device: {DEVICE}...")
        
        # Path where the best model checkpoint is saved
        # Using a relative path starting from the project root
        model_path = os.path.join(os.getcwd(), 'models', 'defect_detector_resnet18.pth')
        
        # Initialize the architecture matching our training setup
        model = get_defect_detection_model(num_classes=len(CLASS_NAMES))
        
        if os.path.exists(model_path):
            # Load weights safely onto the available device
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}. Please train the model first. Using uninitialized weights.")
        
        model = model.to(DEVICE)
        model.eval()  # Set to evaluation mode for inference
        MODEL = model

def get_preprocessing_transforms() -> transforms.Compose:
    """
    Returns the exact torchvision transforms used during training
    to ensure the input data distribution matches what the model expects.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def process_prediction(image_bytes: bytes) -> PredictionResponse:
    """
    Processes the raw bytes of an uploaded image and returns a prediction.
    
    Args:
        image_bytes: The raw byte content of the image.
        
    Returns:
        A PredictionResponse object containing the class, confidence, and timing.
    """
    if MODEL is None:
        raise RuntimeError("Model is not loaded. Ensure load_model() was called on startup.")
        
    start_time = time.time()
    
    # 1. Load image from bytes
    # Convert to RGB because ResNet expects 3 channels
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Preprocess
    transform = get_preprocessing_transforms()
    # Add a batch dimension (B, C, H, W) since the model expects batched inputs
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # 3. Predict
    with torch.no_grad():
        outputs = MODEL(input_tensor)
        # Convert raw logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get the highest probability and its corresponding index
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    # 4. Format Results
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()
    inference_time = time.time() - start_time
    
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence_score=round(confidence_score, 4),
        inference_time_seconds=round(inference_time, 4)
    )
