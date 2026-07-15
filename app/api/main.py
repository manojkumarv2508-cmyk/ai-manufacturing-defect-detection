from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.predict import load_model, process_prediction
from app.api.schemas import PredictionResponse

# Lifespan context manager runs code before the application starts taking requests,
# and yields control back to FastAPI. Great for one-time setup like loading ML models.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup to avoid loading it per request
    load_model()
    yield
    # Cleanup resources on shutdown (if any)
    pass

app = FastAPI(
    title="AI Manufacturing Defect Detection API",
    description="API for classifying casting components as defective or acceptable.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow the future dashboard (frontend) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    """
    Root endpoint for health checking the API.
    """
    return {
        "status": "online",
        "message": "Welcome to the AI Manufacturing Defect Detection API. POST to /predict to classify images."
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file upload and returns the predicted class (defective or ok)
    along with a confidence score and inference time.
    """
    # Basic validation
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        # Read the file bytes asynchronously
        image_bytes = await file.read()
        
        # Process prediction synchronously (since model inference is fast enough, 
        # though for heavy loads, running in a ThreadPoolExecutor is recommended)
        response = process_prediction(image_bytes)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
