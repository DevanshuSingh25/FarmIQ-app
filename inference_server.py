"""
Crop Disease Detection Inference Server with LAZY LOADING
Models are downloaded from HuggingFace and loaded only when needed
"""
import os
import sys
import io
import time
import logging
import numpy as np
import requests
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models_cache"
MODELS_DIR.mkdir(exist_ok=True)

# Model URLs from environment variables or defaults
PLANT_MODEL_URL = os.getenv(
    "PLANT_MODEL_URL",
    "https://huggingface.co/Devanshu2025/SIH-ML/resolve/main/plant_disease_recog_model_pwp.keras"
)
SOIL_MODEL_URL = os.getenv(
    "SOIL_MODEL_URL",
    "https://huggingface.co/Devanshu2025/SIH-ML/resolve/main/soil_model.keras"
)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model cache
_plant_model: Optional[Any] = None
_soil_model: Optional[Any] = None
_plant_model_loading = False
_soil_model_loading = False

logger.info("=" * 70)
logger.info("CROP DISEASE DETECTION INFERENCE SERVER - LAZY LOADING MODE")
logger.info("=" * 70)
logger.info(f"Working Directory: {os.getcwd()}")
logger.info(f"Models Cache Directory: {MODELS_DIR}")
logger.info(f"Plant Model URL: {PLANT_MODEL_URL}")
logger.info(f"Soil Model URL: {SOIL_MODEL_URL}")
logger.info("=" * 70)

# ============================================================================
# CLASS NAMES DEFINITION
# ============================================================================
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

SOIL_CLASS_NAMES = [
    'Black Soil',
    'Cinder Soil',
    'Laterite Soil',
    'Peat Soil',
    'Yellow Soil'
]

# Soil recommendations mapping
SOIL_RECOMMENDATIONS = {
    "Black Soil": {
        "characteristics": "Clayey, moisture-retentive, rich in lime, iron, magnesia; good for deep-rooted crops.",
        "best_crops": ["Cotton", "Maize", "Wheat", "Sunflower", "Moong/Chickpea"],
        "fertilizer": [
            "Apply 1 bag DAP per acre before sowing.",
            "Use 1–1.5 bags Urea per acre in 2 split doses during crop growth.",
            "Add 2 tractor-trolleys gobar compost to soften soil."
        ],
        "tips": [
            "Black soil cracks; maintain steady moisture.",
            "Use mulching in cotton to reduce evaporation.",
            "Avoid over-watering with tube-well irrigation."
        ],
        "irrigation": "Light, frequent irrigation for cotton/maize; stage-wise irrigation for wheat."
    },
    "Cinder Soil": {
        "characteristics": "Soil derived from volcanic ash/lava, porous, well-drained, rich in minerals but sometimes low in organic matter",
        "best_crops": ["Kinnow", "Grapes", "Cotton", "Sugarcane", "Vegetables"],
        "fertilizer": [
            "Apply 1 bag DAP per acre before sowing.",
            "Use 2–3 bags Urea in split doses.",
            "Apply mulch or wheat straw to conserve moisture."
        ],
        "tips": [
            "Soil is porous; keep soil covered.",
            "Drip irrigation recommended for kinnow and grapes."
        ],
        "irrigation": "Frequent light irrigation with mulching."
    },
    "Laterite Soil": {
        "characteristics": "Red/reddish-brown, iron and aluminum rich, acidic pH, low fertility",
        "best_crops": ["Maize", "Sugarcane", "Mustard", "Groundnut", "Vegetables"],
        "fertilizer": [
            "Apply 1 bag DAP per acre before sowing.",
            "Use 1 bag MOP for sugarcane/vegetables.",
            "Add compost or green manure (sunhemp/dhaincha)."
        ],
        "tips": [
            "Foothill regions—risk of erosion; use contour farming.",
            "Add crop residues to improve soil body."
        ],
        "irrigation": "Regular irrigation; avoid runoff."
    },
    "Peat Soil": {
        "characteristics": "Organic-rich, dark brown/black, acidic, high water retention, low nutrient availability",
        "best_crops": ["Rice", "Potato", "Berseem", "Oat", "Vegetables"],
        "fertilizer": [
            "Apply 1 bag DAP per acre before sowing.",
            "Add cow dung compost.",
            "Mix sand to improve drainage."
        ],
        "tips": [
            "Peat soil stays wet; avoid heavy irrigation.",
            "Ideal for paddy-fodder cropping system."
        ],
        "irrigation": "Light irrigation only; keep moist but not water-logged except for rice."
    },
    "Yellow Soil": {
        "characteristics": "Sandy to sandy-loam texture, low organic matter, acidic pH, low fertility, well-drained",
        "best_crops": ["Maize", "Groundnut", "Sugarcane", "Vegetables", "Pulses"],
        "fertilizer": [
            "Apply 1 bag DAP per acre before sowing.",
            "Use 1.5–2 bags Urea during crop growth.",
            "Add compost or green manure crops."
        ],
        "tips": [
            "Add compost to improve soil life.",
            "Grow dhaincha for 45 days before main crop."
        ],
        "irrigation": "Moderate irrigation; avoid over-watering to prevent soil washing."
    }
}

# ============================================================================
# MODEL DOWNLOAD UTILITIES
# ============================================================================
def download_model_from_url(url: str, local_path: Path, token: str = "") -> bool:
    """
    Download model from HuggingFace URL
    Returns True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading model from: {url}")
        logger.info(f"Target path: {local_path}")
        
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        # Stream download to handle large files
        response = requests.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Model size: {total_size / (1024*1024):.2f} MB")
        
        downloaded = 0
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if int(progress) % 10 == 0:  # Log every 10%
                            logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"✅ Model downloaded successfully: {local_path.stat().st_size / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        import traceback
        traceback.print_exc()
        if local_path.exists():
            local_path.unlink()  # Remove partial download
        return False

def load_keras_model(model_path: Path) -> Any:
    """Load Keras model from file"""
    try:
        logger.info(f"Loading Keras model from: {model_path}")
        model = tf.keras.models.load_model(str(model_path), compile=False)
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Input Shape: {model.input_shape}")
        logger.info(f"   Output Shape: {model.output_shape}")
        
        # Test with dummy input
        test_input = np.random.random((1, 160, 160, 3)).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        logger.info(f"   Test prediction successful: {test_output.shape}")
        
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================================
# LAZY LOADING FUNCTIONS
# ============================================================================
def get_plant_model():
    """Get plant disease model, downloading and loading if necessary"""
    global _plant_model, _plant_model_loading
    
    if _plant_model is not None:
        return _plant_model
    
    if _plant_model_loading:
        raise HTTPException(
            status_code=503,
            detail="Plant model is currently being loaded. Please try again in a moment."
        )
    
    try:
        _plant_model_loading = True
        logger.info("")
        logger.info("=" * 70)
        logger.info("LOADING PLANT DISEASE MODEL (LAZY LOADING)")
        logger.info("=" * 70)
        
        model_filename = "plant_disease_recog_model_pwp.keras"
        local_path = MODELS_DIR / model_filename
        
        # Check if model exists locally
        if not local_path.exists():
            # Try to find in parent directory (for backward compatibility)
            parent_path = BASE_DIR / model_filename
            if parent_path.exists():
                logger.info(f"Found model in parent directory: {parent_path}")
                local_path = parent_path
            else:
                logger.info("Model not found locally, downloading from HuggingFace...")
                if not download_model_from_url(PLANT_MODEL_URL, local_path, HF_TOKEN):
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to download plant model from HuggingFace"
                    )
        else:
            logger.info(f"Using cached model: {local_path}")
        
        # Load the model
        _plant_model = load_keras_model(local_path)
        
        logger.info("=" * 70)
        logger.info("✅ PLANT MODEL READY FOR PREDICTIONS")
        logger.info("=" * 70)
        logger.info("")
        
        return _plant_model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load plant model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load plant model: {str(e)}")
    finally:
        _plant_model_loading = False

def get_soil_model():
    """Get soil classification model, downloading and loading if necessary"""
    global _soil_model, _soil_model_loading
    
    if _soil_model is not None:
        return _soil_model
    
    if _soil_model_loading:
        raise HTTPException(
            status_code=503,
            detail="Soil model is currently being loaded. Please try again in a moment."
        )
    
    try:
        _soil_model_loading = True
        logger.info("")
        logger.info("=" * 70)
        logger.info("LOADING SOIL CLASSIFICATION MODEL (LAZY LOADING)")
        logger.info("=" * 70)
        
        model_filename = "soil_model.keras"
        local_path = MODELS_DIR / model_filename
        
        # Check if model exists locally
        if not local_path.exists():
            # Try to find in parent directory (for backward compatibility)
            parent_path = BASE_DIR / model_filename
            if parent_path.exists():
                logger.info(f"Found model in parent directory: {parent_path}")
                local_path = parent_path
            else:
                logger.info("Model not found locally, downloading from HuggingFace...")
                if not download_model_from_url(SOIL_MODEL_URL, local_path, HF_TOKEN):
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to download soil model from HuggingFace"
                    )
        else:
            logger.info(f"Using cached model: {local_path}")
        
        # Load the model
        _soil_model = load_keras_model(local_path)
        
        logger.info("=" * 70)
        logger.info("✅ SOIL MODEL READY FOR PREDICTIONS")
        logger.info("=" * 70)
        logger.info("")
        
        return _soil_model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load soil model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load soil model: {str(e)}")
    finally:
        _soil_model_loading = False

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
TARGET_SIZE = (160, 160)
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model input
    Uses EfficientNet preprocessing (same as training)
    """
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        logger.info(f"   Original image: {img.size}, mode: {img.mode}")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info(f"   Converted to RGB")
        
        # Resize to model input size
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        logger.info(f"   Resized to: {TARGET_SIZE}")
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.uint8)
        logger.info(f"   Array shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Apply EfficientNet preprocessing
        img_array = preprocess_input(img_array)
        logger.info(f"   Value range (after preprocessing): [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        logger.info(f"   Final shape: {img_array.shape}")
        
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Image preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title="Crop Disease Detection API",
    description="Real-time plant disease detection with lazy-loaded ML models",
    version="3.0.0-lazy"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================
@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict plant disease from uploaded image
    Model is loaded on first request (lazy loading)
    """
    request_id = f"REQ_{int(time.time() * 1000)}"
    start_time = time.time()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"📥 NEW PREDICTION REQUEST: {request_id}")
    logger.info(f"   File: {file.filename}")
    logger.info("-" * 70)
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.error(f"❌ Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Get model (will download/load if needed)
        model = get_plant_model()
        
        # Read and preprocess image
        logger.info("Reading and preprocessing image...")
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        logger.info("Making prediction with plant model...")
        prediction_start = time.time()
        predictions = model.predict(processed_image, verbose=0)
        prediction_time = time.time() - prediction_start
        
        logger.info(f"✅ Prediction completed in {prediction_time:.3f}s")
        
        # Process results
        pred_array = predictions[0].copy()
        
        # Apply softmax if needed
        pred_sum = np.sum(pred_array)
        if abs(pred_sum - 1.0) > 0.01:
            exp_preds = np.exp(pred_array - np.max(pred_array))
            pred_array = exp_preds / np.sum(exp_preds)
        
        # Get top prediction
        top_idx = int(np.argmax(pred_array))
        top_confidence = float(pred_array[top_idx])
        
        if top_idx >= len(CLASS_NAMES):
            top_idx = int(np.argmax(pred_array[:len(CLASS_NAMES)]))
            top_confidence = float(pred_array[top_idx])
        
        class_name = CLASS_NAMES[top_idx]
        
        # Get top 3
        top_3_indices = np.argsort(pred_array[:len(CLASS_NAMES)])[-3:][::-1]
        top_3_predictions = [
            {
                "class": CLASS_NAMES[i],
                "confidence": float(pred_array[i])
            }
            for i in top_3_indices
        ]
        
        # Create response
        total_time = time.time() - start_time
        response = {
            "class_name": class_name,
            "confidence": top_confidence,
            "top_3": top_3_predictions,
            "metadata": {
                "request_id": request_id,
                "timestamp": time.time(),
                "processing_time_ms": round(total_time * 1000, 2),
                "prediction_time_ms": round(prediction_time * 1000, 2),
                "loading_strategy": "lazy_loading",
                "model_source": "huggingface"
            }
        }
        
        logger.info(f"✅ PREDICTION: {class_name} ({top_confidence*100:.2f}%)")
        logger.info("=" * 70)
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ PREDICTION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/soil-predict")
async def predict_soil(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict soil type from uploaded image
    Model is loaded on first request (lazy loading)
    """
    request_id = f"SOIL_REQ_{int(time.time() * 1000)}"
    start_time = time.time()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"📥 NEW SOIL PREDICTION REQUEST: {request_id}")
    logger.info(f"   File: {file.filename}")
    logger.info("-" * 70)
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Get model (will download/load if needed)
        model = get_soil_model()
        
        # Read and preprocess image
        logger.info("Reading and preprocessing image...")
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        logger.info("Making prediction with soil model...")
        prediction_start = time.time()
        predictions = model.predict(processed_image, verbose=0)
        prediction_time = time.time() - prediction_start
        
        logger.info(f"✅ Prediction completed in {prediction_time:.3f}s")
        
        # Process results
        pred_array = predictions[0].copy()
        
        # Apply softmax if needed
        pred_sum = np.sum(pred_array)
        if abs(pred_sum - 1.0) > 0.01:
            exp_preds = np.exp(pred_array - np.max(pred_array))
            pred_array = exp_preds / np.sum(exp_preds)
        
        # Get top prediction
        top_idx = int(np.argmax(pred_array))
        top_confidence = float(pred_array[top_idx])
        soil_label = SOIL_CLASS_NAMES[top_idx]
        
        # Get top 3
        top_3_indices = np.argsort(pred_array)[-3:][::-1]
        top_3_predictions = [
            {"label": SOIL_CLASS_NAMES[i], "confidence": float(pred_array[i])}
            for i in top_3_indices
        ]
        
        # Get recommendations
        recommendations = SOIL_RECOMMENDATIONS.get(soil_label, {})
        
        # Create response
        total_time = time.time() - start_time
        response = {
            "soil_label_raw": soil_label,
            "soil_type": soil_label,
            "confidence": top_confidence,
            "top_3": top_3_predictions,
            "recommendations": recommendations,
            "metadata": {
                "request_id": request_id,
                "timestamp": time.time(),
                "processing_time_ms": round(total_time * 1000, 2),
                "prediction_time_ms": round(prediction_time * 1000, 2),
                "loading_strategy": "lazy_loading",
                "model_source": "huggingface"
            }
        }
        
        logger.info(f"✅ SOIL PREDICTION: {soil_label} ({top_confidence*100:.2f}%)")
        logger.info("=" * 70)
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ SOIL PREDICTION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Soil prediction failed: {str(e)}")

@app.post("/soil-predict-debug")
async def predict_soil_debug(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Debug endpoint for soil prediction with detailed output"""
    try:
        model = get_soil_model()
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image, verbose=0)
        pred_array = predictions[0]
        
        # Apply softmax if needed
        pred_sum = np.sum(pred_array)
        if abs(pred_sum - 1.0) > 0.01:
            exp_preds = np.exp(pred_array - np.max(pred_array))
            pred_array = exp_preds / np.sum(exp_preds)
        
        # Get top 5
        top_5_indices = np.argsort(pred_array)[-5:][::-1]
        top_5 = [
            {
                "rank": i+1,
                "soil_type": SOIL_CLASS_NAMES[idx],
                "confidence": float(pred_array[idx]),
                "confidence_pct": f"{pred_array[idx]*100:.2f}%"
            }
            for i, idx in enumerate(top_5_indices)
        ]
        
        return {
            "debug_mode": True,
            "prediction_sum": float(np.sum(pred_array)),
            "top_5_predictions": top_5,
            "loading_strategy": "lazy_loading"
        }
        
    except Exception as e:
        logger.error(f"Debug prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0.0-lazy",
        "loading_strategy": "lazy_loading",
        "plant_model_loaded": _plant_model is not None,
        "soil_model_loaded": _soil_model is not None,
        "plant_model_url": PLANT_MODEL_URL,
        "soil_model_url": SOIL_MODEL_URL,
        "num_plant_classes": len(CLASS_NAMES),
        "num_soil_classes": len(SOIL_CLASS_NAMES),
        "server_time": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Crop Disease Detection API",
        "version": "3.0.0-lazy",
        "status": "running",
        "loading_strategy": "lazy_loading",
        "description": "Models are downloaded from HuggingFace and loaded on first request",
        "endpoints": {
            "predict": "/predict (POST) - Plant disease detection",
            "soil_predict": "/soil-predict (POST) - Soil classification",
            "health": "/health (GET) - Health check"
        },
        "models": {
            "plant_model_loaded": _plant_model is not None,
            "soil_model_loaded": _soil_model is not None
        }
    }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info("")
    logger.info("=" * 70)
    logger.info("Starting FastAPI server with LAZY LOADING...")
    logger.info(f"Server will be available at: http://0.0.0.0:{port}")
    logger.info("Models will be downloaded on first use")
    logger.info("=" * 70)
    logger.info("")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
