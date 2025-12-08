# FarmIQ ML Backend - Deployment Guide

## Lazy Loading Implementation

This FastAPI server now uses **lazy loading** for ML models to work within Render's free tier memory limits.

### How It Works

1. **Server starts without loading models** - Only core dependencies are loaded at startup
2. **Models download on first request** - When `/predict` or `/soil-predict` is called for the first time:
   - Downloads the required model from HuggingFace
   - Caches it locally in `models_cache/` directory
   - Loads it into memory
3. **Subsequent requests use cached model** - No re-downloading or re-loading needed

### Benefits

- ✅ **Low startup memory** - Server starts quickly without OOM errors
- ✅ **Automatic model downloads** - No need to upload large model files
- ✅ **Memory efficient** - Only load models when actually needed
- ✅ **Perfect for free tier** - Works within Render's 512MB limit

## Deployment Steps for Render

### Option 1: Using Blueprint (Recommended)

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Add lazy loading for ML models"
   git push
   ```

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and configure everything

3. **Environment Variables** (Already configured in render.yaml)
   - `PLANT_MODEL_URL` - HuggingFace URL for plant model
   - `SOIL_MODEL_URL` - HuggingFace URL for soil model
   - `HF_TOKEN` - Your HuggingFace token (optional, leave empty if models are public)

### Option 2: Manual Setup

1. **Create New Web Service**
   - Go to Render Dashboard
   - Click "New" → "Web Service"
   - Connect your repository

2. **Configure Service**
   - **Name**: `farmiq-ml-backend`
   - **Environment**: `Python 3`
   - **Region**: `Oregon (US West)`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn inference_server:app --host 0.0.0.0 --port $PORT`

3. **Add Environment Variables**
   - `PLANT_MODEL_URL` = `https://huggingface.co/Devanshu2025/SIH-ML/resolve/main/plant_disease_recog_model_pwp.keras`
   - `SOIL_MODEL_URL` = `https://huggingface.co/Devanshu2025/SIH-ML/resolve/main/soil_model.keras`
   - `HF_TOKEN` = (your token, or leave empty)

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete

## Testing After Deployment

### 1. Check Health
```bash
curl https://your-app.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "loading_strategy": "lazy_loading",
  "plant_model_loaded": false,
  "soil_model_loaded": false
}
```

### 2. Test Plant Disease Prediction
```bash
curl -X POST https://your-app.onrender.com/predict \
  -F "file=@path/to/plant_image.jpg"
```

First request will:
- Download plant model (~100MB)
- Load it into memory
- Make prediction
- Take 30-60 seconds

Subsequent requests will be fast (2-3 seconds).

### 3. Test Soil Prediction
```bash
curl -X POST https://your-app.onrender.com/soil-predict \
  -F "file=@path/to/soil_image.jpg"
```

## Local Testing

### Without Models (Test Lazy Loading)

1. **Move models out temporarily**
   ```powershell
   mkdir temp_models
   Move-Item *.keras temp_models/
   ```

2. **Start server**
   ```powershell
   python inference_server.py
   ```

3. **Test endpoints** - Models should download automatically

4. **Restore models** (optional)
   ```powershell
   Move-Item temp_models/*.keras .
   ```

### With Local Models (Faster Testing)

Models in the same directory will be used automatically without downloading.

## Frontend Integration

Update your frontend environment variable:
- **Vercel**: `VITE_ML_APP` or `VITE_PREDICTION_API_URL`
- **Value**: `https://your-app.onrender.com` (without trailing slash)

## Monitoring

### Render Dashboard
- Check logs for model download progress
- Monitor memory usage (should stay under 512MB)
- First requests will show download logs

### Expected Log Output

**Startup:**
```
CROP DISEASE DETECTION INFERENCE SERVER - LAZY LOADING MODE
Models will be downloaded on first use
```

**First /predict request:**
```
LOADING PLANT DISEASE MODEL (LAZY LOADING)
Downloading model from: https://huggingface.co/...
Download progress: 10%...50%...100%
Model loaded successfully!
```

## Troubleshooting

### "Out of memory" error
- This shouldn't happen with lazy loading
- Check if both models are being loaded simultaneously
- Verify only one endpoint is called at a time initially

### "Failed to download model"
- Check HuggingFace URLs are correct
- Verify internet connectivity
- Check if HF_TOKEN is needed (for private repos)

### Slow first request
- Normal! Model download takes 30-60 seconds
- Subsequent requests will be fast
- Consider calling both endpoints once after deployment to "warm up"

## Model Update Process

To update models:
1. Upload new model to HuggingFace
2. Update environment variable in Render dashboard
3. Restart web service
4. Models will re-download on next request

## Memory Usage

- **Startup**: ~100MB (Python + FastAPI + TensorFlow)
- **With plant model**: ~250MB
- **With soil model**: ~250MB
- **Both models**: ~400MB (still within 512MB limit)

Lazy loading ensures you never exceed memory limits!
