from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    """
    Schema for the response returned by the /predict endpoint.
    """
    predicted_class: str
    confidence_score: float
    inference_time_seconds: float
    message: Optional[str] = "Success"
