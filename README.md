# FarmIQ - Agricultural Technology Platform

## Quick Start

### Step 1: Clone the repository
```bash
git clone <YOUR_GIT_URL>
cd <YOUR_PROJECT_NAME>
```

### Step 2: Install Frontend Dependencies
```bash
npm install
```

### Step 3: Install Backend Dependencies (Node.js)
```bash
cd server
npm install
cd ..
```

### Step 4: Setup Python Inference Server (for Crop Disease Detection)

1. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the inference server:
```bash
uvicorn inference_server:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5: Start Backend Server (Node.js)
```bash
cd server
node server.js
# Server runs on http://localhost:3001
```

### Step 6: Start Frontend Development Server
```bash
npm run dev
# Frontend runs on http://localhost:5173
```

## Environment Variables

Copy `env.example` to `.env` and configure:

```bash
VITE_API_URL=http://localhost:3001/api
VITE_PREDICTION_API_URL=http://localhost:8000
```

## Features

- **Authentication**: Multi-role authentication (Farmer, Vendor, Admin)
- **Weather Service**: 7-day forecasts and farming advice
- **IoT Sensor Management**: Installation requests and live readings
- **Market Prices**: Real-time commodity prices with filtering
- **Crop Disease Detection**: AI-powered disease identification using Keras model
- **Soil Analysis**: Soil health analysis
- **NGO Schemes**: Government scheme information
- And more...

## Crop Disease Detection

The crop disease detection feature uses a trained Keras model (`plant_disease_recog_model_pwp.keras`) to identify plant diseases from uploaded images.

**Important**: The inference server must be running for disease detection to work. See `INFERENCE_SERVER_README.md` for detailed setup instructions.

## Project Structure

```
FarmIQ-login/
├── src/                    # Frontend React/TypeScript code
│   ├── pages/             # Page components
│   ├── components/        # Reusable components
│   ├── services/         # API service clients
│   └── utils/            # Utility functions
├── server/                # Node.js backend
│   ├── server.js         # Express server
│   ├── auth.js           # Authentication logic
│   └── database.js       # SQLite database
├── inference_server.py   # FastAPI inference server (Python)
├── plant_disease_recog_model_pwp.keras  # Trained Keras model
└── requirements.txt      # Python dependencies
```

## Documentation

- **Tech Stack**: See `TECH_STACK.md` for complete technology stack and architecture
- **Inference Server**: See `INFERENCE_SERVER_README.md` for detailed setup and API documentation
- **Authentication**: See `AUTH_README.md` for authentication flow
- **Deployment**: See `deployment-guide.md` for production deployment
- **Troubleshooting**: See `TROUBLESHOOTING_LOGIN.md` for login issues
- **Implementation**: See `IMPLEMENTATION_CHECKLIST.md` for testing checklist

## Testing

### Test Inference Server
```bash
# Test prediction endpoint
curl -F "file=@sample.jpg" http://localhost:8000/predict
```

Expected response:
```json
{"class_name":"Tomato___Late_blight","confidence":0.87}
```

### Test Health Endpoint
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","model_loaded":true}
```

## Troubleshooting

### Crop Disease Detection Not Working
1. Ensure inference server is running on port 8000
2. Check browser console for CORS errors
3. Verify model file exists: `plant_disease_recog_model_pwp.keras`
4. See `INFERENCE_SERVER_README.md` for detailed troubleshooting

### CORS Errors
- Add your frontend origin to `ALLOWED_ORIGINS` in `inference_server.py`
- Ensure backend CORS is configured correctly in `server/server.js`

## License

[Your License Here] 
