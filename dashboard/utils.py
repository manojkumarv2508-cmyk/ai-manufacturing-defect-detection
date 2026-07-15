import requests
import streamlit as st
import os
from typing import Dict, Any, Optional

# Allow API URL to be configured via environment variable, fallback to localhost
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

def check_api_health() -> bool:
    """Checks if the FastAPI backend is reachable."""
    try:
        # Check the root endpoint for health
        health_url = API_URL.replace("/predict", "/")
        response = requests.get(health_url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def predict_image(image_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """
    Sends an image to the FastAPI backend and returns the prediction result.
    
    Args:
        image_bytes: Raw bytes of the image file.
        filename: Name of the uploaded file.
        
    Returns:
        Dictionary containing the API response, or None if an error occurred.
    """
    try:
        # Formulate the multipart/form-data request
        files = {"file": (filename, image_bytes, "image/jpeg")}
        
        # Send POST request to FastAPI backend
        response = requests.post(API_URL, files=files, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}) for {filename}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error(f"Request timeout for {filename}. The backend API took too long to respond.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend API. Please ensure the FastAPI server is running on http://127.0.0.1:8000")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def format_confidence(confidence: float) -> str:
    """Formats confidence score as a percentage string."""
    return f"{confidence * 100:.2f}%"

def format_inference_time(time_seconds: float) -> str:
    """Formats inference time from seconds to milliseconds string."""
    return f"{time_seconds * 1000:.1f} ms"

def get_status_color(prediction_class: str) -> str:
    """
    Returns a color hex code based on the prediction class for Plotly charts.
    Matches standard industrial UI colors (Red for defective, Green for OK).
    """
    if "def" in prediction_class.lower():
        return "#FF4B4B" # Streamlit Red
    return "#00CC96" # Plotly Green
