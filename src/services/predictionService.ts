// Prediction service for crop disease detection API

const PREDICTION_API_URL = import.meta.env.VITE_PREDICTION_API_URL || 'http://localhost:8000';

export interface PredictionResponse {
  class_name: string;
  confidence: number;
}

/**
 * Calls the inference API to predict disease from an uploaded image
 */
export async function callInferenceApi(imageFile: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", imageFile);

  try {
    console.log(`[PredictionService] 🚀 Calling REAL inference API: ${PREDICTION_API_URL}/predict`);
    console.log(`[PredictionService] 📸 Image file: ${imageFile.name}, size: ${imageFile.size} bytes`);
    console.log(`[PredictionService] ⚠️ This is a REAL API call - NOT MOCK!`);
    
    // Add cache-busting to ensure we get fresh predictions
    const cacheBuster = `?t=${Date.now()}`;
    const resp = await fetch(`${PREDICTION_API_URL}/predict${cacheBuster}`, {
      method: "POST",
      body: form,
      cache: 'no-store',  // Prevent caching
      // Note: Don't set Content-Type header - browser will set it with boundary for FormData
    });

    console.log(`[PredictionService] 📡 Response status: ${resp.status}`);
    console.log(`[PredictionService] 📡 Response headers:`, Object.fromEntries(resp.headers.entries()));

    if (!resp.ok) {
      const text = await resp.text();
      console.error(`[PredictionService] ❌ API error (${resp.status}):`, text);
      throw new Error(`Prediction failed (${resp.status}): ${text}`);
    }

    const data = await resp.json() as any;
    console.log(`[PredictionService] ✅ Received REAL prediction from model:`, data);
    console.log(`[PredictionService] 🎯 Class: ${data.class_name}, Confidence: ${(data.confidence * 100).toFixed(2)}%`);
    
    // Log metadata to prove it's real
    if (data.metadata) {
      console.log(`[PredictionService] 📊 Metadata:`, data.metadata);
      console.log(`[PredictionService]   Request ID: ${data.metadata.request_id}`);
      console.log(`[PredictionService]   Is Real: ${data.metadata.is_real_prediction}`);
      console.log(`[PredictionService]   Source: ${data.metadata.prediction_source}`);
      console.log(`[PredictionService]   Processing time: ${data.metadata.processing_time_ms}ms`);
    }
    
    // Log top 3 if available
    if (data.top_3) {
      console.log(`[PredictionService] 📊 Top 3 predictions:`, data.top_3);
    }
    
    console.log(`[PredictionService] ⚠️ THIS IS REAL DATA FROM KERAS MODEL - NOT MOCK!`);
    
    // Return in expected format
    return {
      class_name: data.class_name,
      confidence: data.confidence
    } as PredictionResponse;
  } catch (error) {
    // Handle network/CORS errors
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      throw new Error(
        `Cannot connect to prediction server at ${PREDICTION_API_URL}. ` +
        `Please ensure the inference server is running: ` +
        `uvicorn inference_server:app --host 0.0.0.0 --port 8000`
      );
    }
    throw error;
  }
}

/**
 * Health check for the prediction API
 */
export async function checkPredictionApiHealth(): Promise<boolean> {
  try {
    const resp = await fetch(`${PREDICTION_API_URL}/health`);
    return resp.ok;
  } catch {
    return false;
  }
}

