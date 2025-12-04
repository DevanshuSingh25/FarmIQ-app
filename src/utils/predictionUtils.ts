// Utility functions for disease prediction mapping and status computation

export interface DiseaseInfo {
  title: string;
  cause: string;
  treatment: string;
}

/**
 * Maps predicted class name to user-friendly disease information
 */
export function getDiseaseInfo(predictedClass: string): DiseaseInfo {
  // Handle background/no leaves case
  if (predictedClass === "Background_without_leaves") {
    return {
      title: "No Disease Detected",
      cause: "The image does not contain clear plant leaves. Please upload a clear image of the affected plant part.",
      treatment: "Take a new photo with good lighting, focusing on the affected leaves or plant parts."
    };
  }

  // Handle healthy cases
  if (predictedClass.includes("healthy")) {
    const crop = predictedClass.split("___")[0];
    return {
      title: `${crop} - Healthy`,
      cause: "No disease symptoms detected. The plant appears to be in good health.",
      treatment: "Continue current care practices. Monitor regularly for any changes."
    };
  }

  // Disease mapping - comprehensive mapping for all classes
  const diseaseMap: Record<string, DiseaseInfo> = {
    // Apple diseases
    "Apple___Apple_scab": {
      title: "Apple Scab",
      cause: "Fungal disease caused by Venturia inaequalis. Thrives in cool, wet conditions during spring and early summer.",
      treatment: "Apply fungicides containing captan or mancozeb. Remove fallen leaves in autumn. Plant resistant varieties. Improve air circulation."
    },
    "Apple___Black_rot": {
      title: "Apple Black Rot",
      cause: "Fungal disease caused by Botryosphaeria obtusa. Spreads through infected plant material and wounds.",
      treatment: "Prune infected branches. Apply copper-based fungicides. Remove mummified fruits. Maintain tree health."
    },
    "Apple___Cedar_apple_rust": {
      title: "Cedar Apple Rust",
      cause: "Fungal disease requiring both apple and cedar trees. Caused by Gymnosporangium juniperi-virginianae.",
      treatment: "Remove nearby cedar trees if possible. Apply fungicides in early spring. Plant resistant varieties."
    },
    
    // Corn diseases
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
      title: "Corn Gray Leaf Spot",
      cause: "Fungal disease caused by Cercospora zeae-maydis. Favors warm, humid conditions.",
      treatment: "Apply fungicides containing azoxystrobin or propiconazole. Rotate crops. Use resistant hybrids."
    },
    "Corn___Common_rust": {
      title: "Corn Common Rust",
      cause: "Fungal disease caused by Puccinia sorghi. Spreads through wind-borne spores.",
      treatment: "Apply fungicides early in season. Plant resistant varieties. Remove crop residue after harvest."
    },
    "Corn___Northern_Leaf_Blight": {
      title: "Corn Northern Leaf Blight",
      cause: "Fungal disease caused by Exserohilum turcicum. Thrives in cool, wet weather.",
      treatment: "Apply fungicides containing azoxystrobin. Rotate crops. Use resistant hybrids. Remove infected debris."
    },
    
    // Grape diseases
    "Grape___Black_rot": {
      title: "Grape Black Rot",
      cause: "Fungal disease caused by Guignardia bidwellii. Spreads in warm, humid conditions.",
      treatment: "Apply fungicides containing mancozeb or captan. Remove infected berries and leaves. Improve air circulation."
    },
    "Grape___Esca_(Black_Measles)": {
      title: "Grape Esca (Black Measles)",
      cause: "Complex disease caused by multiple fungi. Often affects older vines.",
      treatment: "Prune infected wood. Apply fungicides. Maintain vine health. Remove severely affected vines."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
      title: "Grape Leaf Blight",
      cause: "Fungal disease caused by Isariopsis griseola. Favors warm, humid conditions.",
      treatment: "Apply copper-based fungicides. Remove infected leaves. Improve air circulation."
    },
    
    // Peach diseases
    "Peach___Bacterial_spot": {
      title: "Peach Bacterial Spot",
      cause: "Bacterial disease caused by Xanthomonas arboricola. Spreads in warm, wet conditions.",
      treatment: "Apply copper-based bactericides. Prune infected branches. Avoid overhead irrigation. Plant resistant varieties."
    },
    
    // Pepper diseases
    "Pepper,_bell___Bacterial_spot": {
      title: "Bell Pepper Bacterial Spot",
      cause: "Bacterial disease caused by Xanthomonas species. Spreads through water and infected seeds.",
      treatment: "Use disease-free seeds. Apply copper-based bactericides. Avoid overhead watering. Rotate crops."
    },
    
    // Potato diseases
    "Potato___Early_blight": {
      title: "Potato Early Blight",
      cause: "Fungal disease caused by Alternaria solani. Common in warm, humid conditions.",
      treatment: "Apply fungicides containing chlorothalonil or mancozeb. Rotate crops. Remove infected foliage. Improve spacing."
    },
    "Potato___Late_blight": {
      title: "Potato Late Blight",
      cause: "Fungal disease caused by Phytophthora infestans. Devastating disease that spreads rapidly in cool, wet conditions.",
      treatment: "Apply fungicides immediately (mancozeb, chlorothalonil). Remove and destroy infected plants. Avoid overhead irrigation. Use certified seed potatoes."
    },
    
    // Squash diseases
    "Squash___Powdery_mildew": {
      title: "Squash Powdery Mildew",
      cause: "Fungal disease caused by Podosphaera xanthii. Common in warm, dry conditions with high humidity.",
      treatment: "Apply fungicides containing sulfur or potassium bicarbonate. Improve air circulation. Remove infected leaves. Plant resistant varieties."
    },
    
    // Strawberry diseases
    "Strawberry___Leaf_scorch": {
      title: "Strawberry Leaf Scorch",
      cause: "Fungal disease caused by Diplocarpon earliana. Thrives in warm, humid conditions.",
      treatment: "Apply fungicides containing captan or thiophanate-methyl. Remove infected leaves. Improve air circulation. Use resistant varieties."
    },
    
    // Cherry diseases
    "Cherry___Powdery_mildew": {
      title: "Cherry Powdery Mildew",
      cause: "Fungal disease caused by Podosphaera clandestina. Favors warm, dry conditions.",
      treatment: "Apply fungicides containing sulfur or myclobutanil. Prune for better air circulation. Remove infected leaves."
    },
    
    // Tomato diseases
    "Tomato___Bacterial_spot": {
      title: "Tomato Bacterial Spot",
      cause: "Bacterial disease caused by Xanthomonas species. Spreads through water, tools, and infected seeds.",
      treatment: "Use disease-free seeds. Apply copper-based bactericides. Avoid overhead watering. Rotate crops. Remove infected plants."
    },
    "Tomato___Early_blight": {
      title: "Tomato Early Blight",
      cause: "Fungal disease caused by Alternaria solani. Common in warm, humid weather.",
      treatment: "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves. Improve spacing. Rotate crops."
    },
    "Tomato___Late_blight": {
      title: "Tomato Late Blight",
      cause: "Fungal disease caused by Phytophthora infestans. Devastating disease that spreads rapidly in cool, wet conditions.",
      treatment: "Apply fungicides immediately (mancozeb, chlorothalonil). Remove and destroy infected plants. Avoid overhead irrigation. Improve air circulation."
    },
    "Tomato___Leaf_Mold": {
      title: "Tomato Leaf Mold",
      cause: "Fungal disease caused by Passalora fulva. Thrives in high humidity and moderate temperatures.",
      treatment: "Improve air circulation. Reduce humidity. Apply fungicides containing chlorothalonil. Remove infected leaves."
    },
    "Tomato___Septoria_leaf_spot": {
      title: "Tomato Septoria Leaf Spot",
      cause: "Fungal disease caused by Septoria lycopersici. Spreads through water splashes and infected debris.",
      treatment: "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves. Avoid overhead watering. Rotate crops."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
      title: "Tomato Spider Mites",
      cause: "Pest infestation by two-spotted spider mites (Tetranychus urticae). Thrives in hot, dry conditions.",
      treatment: "Apply miticides or insecticidal soap. Increase humidity. Remove heavily infested leaves. Use predatory mites for biological control."
    },
    "Tomato___Target_Spot": {
      title: "Tomato Target Spot",
      cause: "Fungal disease caused by Corynespora cassiicola. Favors warm, humid conditions.",
      treatment: "Apply fungicides containing azoxystrobin or chlorothalonil. Remove infected leaves. Improve air circulation."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
      title: "Tomato Yellow Leaf Curl Virus",
      cause: "Viral disease transmitted by whiteflies (Bemisia tabaci). No cure once infected.",
      treatment: "Remove and destroy infected plants. Control whitefly populations with insecticides. Use resistant varieties. Use row covers."
    },
    "Tomato___Tomato_mosaic_virus": {
      title: "Tomato Mosaic Virus",
      cause: "Viral disease spread through contact, tools, and infected seeds. Very contagious.",
      treatment: "Remove and destroy infected plants. Disinfect tools. Use disease-free seeds. Avoid handling plants when wet."
    }
  };

  // Return mapped disease info or default
  return diseaseMap[predictedClass] || {
    title: predictedClass.replace(/_/g, " ").replace("___", " - "),
    cause: "Disease information not available. Please consult with an agricultural expert for specific diagnosis and treatment.",
    treatment: "Consult with a local agricultural extension service or plant pathologist for accurate diagnosis and treatment recommendations."
  };
}

/**
 * Computes detection status based on predicted class
 */
export function computeDetectionStatus(predictedClass: string | null | undefined): "no detected" | "Positive" | "Not available" {
  if (!predictedClass) return "Not available";
  
  if (predictedClass === "Background_without_leaves") {
    return "no detected";
  }
  
  if (predictedClass.includes("healthy")) {
    return "no detected";
  }
  
  return "Positive";
}

