import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, Any

def init_history():
    """
    Initializes the history dataframe in Streamlit's session state 
    if it doesn't already exist. This ensures data persists across reruns.
    """
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(
            columns=['Timestamp', 'Filename', 'Prediction', 'Confidence', 'Inference Time (s)']
        )

def add_prediction(filename: str, result: Dict[str, Any]):
    """
    Adds a new prediction result to the history dataframe.
    
    Args:
        filename: Name of the processed image file.
        result: The JSON response from the API containing the prediction metrics.
    """
    init_history()
    
    new_entry = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Filename': filename,
        'Prediction': result.get('predicted_class', 'Unknown'),
        'Confidence': result.get('confidence_score', 0.0),
        'Inference Time (s)': result.get('inference_time_seconds', 0.0)
    }
    
    # Use pd.concat to add the new row to the top of the history
    st.session_state.prediction_history = pd.concat(
        [pd.DataFrame([new_entry]), st.session_state.prediction_history],
        ignore_index=True
    )

def get_history() -> pd.DataFrame:
    """
    Returns the current prediction history dataframe.
    """
    init_history()
    return st.session_state.prediction_history

def clear_history():
    """
    Clears the prediction history, resetting the dataframe.
    """
    st.session_state.prediction_history = pd.DataFrame(
        columns=['Timestamp', 'Filename', 'Prediction', 'Confidence', 'Inference Time (s)']
    )
