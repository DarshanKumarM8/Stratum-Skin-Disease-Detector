from flask import Flask, request, Response, jsonify
from flask.templating import render_template
from flask import request
from werkzeug.utils import secure_filename
from app import app
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
import os

# --- KNOWLEDGE BASE FOR RECOMMENDATIONS ---
KNOWLEDGE_BASE = {
    "Oily": {
        "Acne": {"risk": "High", "advice": "Oily skin exacerbates Acne. Use Salicylic acid cleanser and oil-free products.", "urgent": False},
        "Acne Scars": {"risk": "Moderate", "advice": "Use vitamin C serum and retinoids. Consider chemical peels for severe scarring.", "urgent": False},
        "Eczema": {"risk": "Moderate", "advice": "Unusual for oily skin. Check for allergic reaction or irritants.", "urgent": True},
        "Psoriasis": {"risk": "Moderate", "advice": "Keep area clean. Avoid heavy oils and use gentle cleansers.", "urgent": True},
        "Melanoma": {"risk": "Critical", "advice": "Immediate dermatologist consultation required. Do not delay.", "urgent": True},
        "Vitiligo": {"risk": "Moderate", "advice": "Consult a dermatologist for phototherapy options. Use sunscreen on affected areas.", "urgent": False},
        "Warts": {"risk": "Low", "advice": "Common viral infection. Use salicylic acid treatment or see dermatologist for cryotherapy.", "urgent": False},
        "Melasma": {"risk": "Moderate", "advice": "Use broad-spectrum sunscreen daily. Consider hydroquinone or vitamin C serums.", "urgent": False},
        "Acanthosis Nigricans": {"risk": "High", "advice": "May indicate insulin resistance. Check blood sugar levels and consult endocrinologist.", "urgent": True},
        "Alopecia Areata": {"risk": "Moderate", "advice": "Autoimmune condition. Consult dermatologist for corticosteroid treatments.", "urgent": False},
        "default": {"risk": "Moderate", "advice": "Monitor the condition. Use oil-control products and stay hydrated.", "urgent": False}
    },
    "Dry": {
        "Acne": {"risk": "Low", "advice": "Likely hormonal. Use gentle hydration, avoid drying agents.", "urgent": False},
        "Acne Scars": {"risk": "Moderate", "advice": "Keep skin hydrated. Use gentle exfoliation and healing serums.", "urgent": False},
        "Eczema": {"risk": "Critical", "advice": "Dry skin triggers Eczema flare-ups. Intense hydration needed.", "urgent": True},
        "Psoriasis": {"risk": "High", "advice": "Dryness worsens plaques. Use urea-based moisturizers.", "urgent": True},
        "Melanoma": {"risk": "Critical", "advice": "Immediate dermatologist consultation required. Do not delay.", "urgent": True},
        "Vitiligo": {"risk": "Moderate", "advice": "Keep skin moisturized. Use sunscreen and consult dermatologist for treatment options.", "urgent": False},
        "Warts": {"risk": "Low", "advice": "Keep area moisturized. Use OTC wart treatments or consult dermatologist.", "urgent": False},
        "Melasma": {"risk": "Moderate", "advice": "Protect from sun exposure. Use gentle, hydrating sunscreens.", "urgent": False},
        "Acanthosis Nigricans": {"risk": "High", "advice": "May indicate metabolic disorder. Consult doctor for underlying causes.", "urgent": True},
        "Alopecia Areata": {"risk": "Moderate", "advice": "Autoimmune condition affecting hair. Seek dermatologist consultation.", "urgent": False},
        "default": {"risk": "Moderate", "advice": "Keep skin moisturized. Use gentle, fragrance-free products.", "urgent": False}
    },
    "Normal": {
        "Acne": {"risk": "Low", "advice": "Maintain good skincare routine. Use gentle cleansers.", "urgent": False},
        "Acne Scars": {"risk": "Low", "advice": "Consider retinoid treatments or professional procedures.", "urgent": False},
        "Vitiligo": {"risk": "Moderate", "advice": "Consult dermatologist. Phototherapy and topical treatments available.", "urgent": False},
        "Warts": {"risk": "Low", "advice": "Common condition. OTC treatments available or see dermatologist.", "urgent": False},
        "Melasma": {"risk": "Low", "advice": "Use sunscreen daily. Topical treatments can help fade pigmentation.", "urgent": False},
        "Acanthosis Nigricans": {"risk": "High", "advice": "Check for insulin resistance or hormonal issues.", "urgent": True},
        "Alopecia Areata": {"risk": "Moderate", "advice": "Consult dermatologist for treatment options.", "urgent": False},
        "default": {"risk": "Moderate", "advice": "Consult a dermatologist for specific diagnosis and treatment.", "urgent": False}
    }
}

# Classes for your local skin model
SKIN_CLASSES = ['acanthosis-nigricans', 'acne', 'acne-scars', 'alopecia-areata', 
                'dry', 'melasma', 'oily', 'vitiligo', 'warts']

# Skin type only classes (not diseases)
SKIN_TYPE_ONLY = ['dry', 'oily']

# Disease conditions that local model can detect
DISEASE_CONDITIONS = ['acanthosis-nigricans', 'acne', 'acne-scars', 'alopecia-areata', 
                      'melasma', 'vitiligo', 'warts']

# Global model variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Pre-load models at startup
print("=" * 50)
print("LOADING MODELS AT STARTUP...")
print("=" * 50)

# Load local model
print("Loading local skin model...")
try:
    local_model = torch.load('./skin-model-pokemon.pt', map_location=device, weights_only=False)
    local_model.to(device)
    local_model.eval()
    print("✓ Local skin model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load local model: {e}")
    local_model = None

# Load HuggingFace model
print("Loading Hugging Face disease model...")
try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    hf_model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    hf_processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    hf_model.eval()
    print("✓ Hugging Face model loaded successfully!")
    print(f"  Available labels: {hf_model.config.id2label}")
except Exception as e:
    print(f"✗ Failed to load HuggingFace model: {e}")
    hf_model = None
    hf_processor = None

print("=" * 50)
print("MODEL LOADING COMPLETE")
print("=" * 50)

def get_transforms():
    transform = []
    transform.append(T.Resize((512, 512)))
    transform.append(T.ToTensor())
    return T.Compose(transform)

def predict_local(model, img, tr, classes):
    """Predict using local model"""
    if model is None:
        print("ERROR: Local model is None!")
        return "unknown", 0.0
    
    img_tensor = tr(img)
    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
    
    probs = F.softmax(out, dim=1)
    print(f"[LOCAL MODEL] Raw probabilities: {probs}")
    
    # Print all class probabilities for debugging
    for i, (cls, prob) in enumerate(zip(classes, probs[0])):
        print(f"  {cls}: {prob.item()*100:.2f}%")
    
    confidence, idx = torch.max(probs, 1)
    result = classes[idx.item()]
    print(f"[LOCAL MODEL] Prediction: {result} ({confidence.item()*100:.1f}%)")
    return result, confidence.item()

def predict_disease_hf(image_path):
    """Predict disease using Hugging Face model"""
    global hf_model, hf_processor
    
    print(f"[HF MODEL] Starting prediction for: {image_path}")
    print(f"[HF MODEL] Model loaded: {hf_model is not None}, Processor loaded: {hf_processor is not None}")
    
    if hf_model is None or hf_processor is None:
        print("[HF MODEL] Models not available!")
        return "Model not loaded", 0.0
    
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"[HF MODEL] Image loaded: {img.size}")
        
        inputs = hf_processor(img, return_tensors="pt")
        print(f"[HF MODEL] Inputs prepared")
        
        with torch.no_grad():
            outputs = hf_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probs, 1)
        
        labels = hf_model.config.id2label
        disease_name = labels[predicted_class.item()]
        
        print(f"[HF MODEL] Raw prediction: {disease_name} ({confidence.item()*100:.1f}%)")
        
        # Make the name more readable
        simplified_name = disease_name.replace("_", " ").title()
        
        print(f"[HF MODEL] Final prediction: {simplified_name}")
        return simplified_name, confidence.item()
        
    except Exception as e:
        import traceback
        print(f"[HF MODEL] ERROR: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", 0.0

def get_skin_type(condition):
    """Determine skin type from local model prediction"""
    if condition.lower() == 'oily':
        return 'Oily'
    elif condition.lower() == 'dry':
        return 'Dry'
    else:
        return 'Normal'

def generate_analysis(skin_condition, skin_type, disease_name, disease_confidence):
    """Generate comprehensive analysis report"""
    
    # Calculate intensity score (0-10) based on confidence
    local_intensity = 7.0  # Default for local model
    
    if disease_name and disease_confidence > 0:
        hf_intensity = round(disease_confidence * 10, 1)
        overall_intensity = round((local_intensity + hf_intensity) / 2, 1)
    else:
        hf_intensity = 0
        overall_intensity = local_intensity
    
    # Get recommendation from knowledge base
    disease_key = disease_name if disease_name else skin_condition.title()
    kb_entry = KNOWLEDGE_BASE.get(skin_type, KNOWLEDGE_BASE["Normal"]).get(
        disease_key, 
        KNOWLEDGE_BASE.get(skin_type, KNOWLEDGE_BASE["Normal"]).get("default")
    )
    
    analysis = {
        "skin_condition": skin_condition,
        "skin_type": skin_type,
        "disease_detected": disease_name if disease_name else "Not available",
        "disease_confidence": round(disease_confidence * 100, 1) if disease_confidence else 0,
        "intensity_score": overall_intensity,
        "risk_level": kb_entry["risk"],
        "recommendation": kb_entry["advice"],
        "urgent": kb_entry["urgent"] or overall_intensity > 8.5
    }
    
    return analysis

@app.route('/', methods=['GET', 'POST'])
def home_page():
    res = None
    analysis = None
    
    if request.method == 'POST':
        print("\n" + "=" * 50)
        print("NEW PREDICTION REQUEST")
        print("=" * 50)
        
        f = request.files['file']
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_PATH'], filename)
        f.save(path)
        print(f"Image saved to: {path}")
        
        # Get transforms
        tr = get_transforms()
        img = Image.open(path).convert("RGB")
        print(f"Image size: {img.size}")
        
        # Step 1: Predict using local model
        print("\n--- Step 1: Local Model Prediction ---")
        skin_condition, local_confidence = predict_local(local_model, img, tr, SKIN_CLASSES)
        skin_type = get_skin_type(skin_condition)
        print(f"Skin condition: {skin_condition}, Skin type: {skin_type}")
        
        # Step 2: Run HF model for cancer screening
        print("\n--- Step 2: HuggingFace Cancer Screening ---")
        hf_disease_name, hf_confidence = predict_disease_hf(path)
        print(f"HF Result: {hf_disease_name} ({hf_confidence*100:.1f}%)")
        
        # Step 3: Combine results from both models
        print("\n--- Step 3: Combining Model Results ---")
        
        # Check if local model detected a specific disease (not just skin type)
        local_is_disease = skin_condition.lower() not in SKIN_TYPE_ONLY
        
        # Check if HF model detected a TRUE cancer condition (melanoma specifically)
        # Actinic Keratoses is often a false positive, so we only prioritize melanoma
        high_risk_cancers = ['melanoma', 'basal cell carcinoma']
        hf_is_high_risk = any(cancer.lower() in hf_disease_name.lower() for cancer in high_risk_cancers)
        
        # Decision logic:
        # 1. If HF detected high-risk cancer (melanoma) with very high confidence (>70%), prioritize it
        # 2. If local model detected a specific disease (vitiligo, acne, etc.), use it
        # 3. If local model detected skin type with good confidence, trust it
        # 4. Only use HF for skin type cases if local confidence is very low
        
        if hf_is_high_risk and hf_confidence > 0.7:
            # Very high confidence cancer detection - prioritize this
            disease_name = hf_disease_name
            disease_confidence = hf_confidence
            print(f"⚠️ CANCER WARNING: Using HF model: {disease_name} ({disease_confidence*100:.1f}%)")
        elif local_is_disease:
            # Local model found a specific condition (vitiligo, acne, etc.)
            disease_name = skin_condition.replace('-', ' ').title()
            disease_confidence = local_confidence
            print(f"Using LOCAL model: {disease_name} ({disease_confidence*100:.1f}%)")
        else:
            # Local model detected skin type (dry/oily) - trust it
            # Only use HF if local confidence is very low
            if local_confidence < 0.4 and hf_confidence > 0.7:
                disease_name = hf_disease_name
                disease_confidence = hf_confidence
                print(f"Low local confidence, using HF: {disease_name} ({disease_confidence*100:.1f}%)")
            else:
                disease_name = skin_condition.replace('-', ' ').title()
                disease_confidence = local_confidence
                print(f"Using LOCAL model: {disease_name} ({disease_confidence*100:.1f}%)")
        
        # Step 4: Generate analysis
        print("\n--- Step 4: Generating Analysis ---")
        analysis = generate_analysis(skin_condition, skin_type, disease_name, disease_confidence)
        print(f"Analysis: {analysis}")
        
        res = skin_condition
        print("\n" + "=" * 50)
        print("PREDICTION COMPLETE")
        print("=" * 50)

    return render_template("index.html", res=res, analysis=analysis)
