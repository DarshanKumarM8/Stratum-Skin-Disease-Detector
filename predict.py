import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import AutoModelForImageClassification

# --- CONFIGURATION ---
# 1. Load Your Local Skin Type Model (The one you already have)
# Ensure 'skin-model-pokemon.pt' is in the same folder
SKIN_TYPE_MODEL_PATH = "skin-model-pokemon.pt"

# 2. Define the Correlation & Risk Logic
# This dictionary maps (Skin Type + Disease) -> Recommendation & Risk Level
KNOWLEDGE_BASE = {
    "Oily": {
        "Acne": {"risk": "High", "advice": "Oily skin exacerbates Acne. Use Salicylic acid cleanser.", "urgent": False},
        "Eczema": {"risk": "Moderate", "advice": "Unusual for oily skin. Check for allergic reaction.", "urgent": True},
        "Psoriasis": {"risk": "Moderate", "advice": "Keep area clean. Avoid heavy oils.", "urgent": True},
    },
    "Dry": {
        "Acne": {"risk": "Low", "advice": "Likely hormonal. Use gentle hydration, avoid drying agents.", "urgent": False},
        "Eczema": {"risk": "Critical", "advice": "Dry skin triggers Eczema flare-ups. Intense hydration needed.", "urgent": True},
        "Psoriasis": {"risk": "High", "advice": "Dryness worsens plaques. Use urea-based moisturizers.", "urgent": True},
    }
}

def load_models():
    """Loads both the local skin type model and the remote disease model."""
    print("Loading Skin Type Model...")
    # Loading your existing local model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skin_type_model = torch.load(SKIN_TYPE_MODEL_PATH, map_location=device, weights_only=False)
    skin_type_model.eval()

    print("Loading Disease Detection Model (from Hugging Face)...")
    # Using a pre-trained model for Acne, Eczema, etc.
    disease_model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    disease_model.eval()
    
    return skin_type_model, disease_model

def predict_skin_type(model, image_path):
    """Predicts Oily vs Dry using your local model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        
    # Assuming Index 0 = Dry, Index 1 = Oily (Update if your training was different!)
    types = ["Dry", "Oily"] 
    return types[predicted.item()]

def predict_disease(model, image_path):
    """Predicts disease (Acne, Eczema, etc.) using Hugging Face model."""
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
    
    img = Image.open(image_path).convert('RGB')
    inputs = processor(img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probs, 1)

    # Get the label name from the model's config
    labels = model.config.id2label
    disease_name = labels[predicted_class.item()]
    
    # Simplify names for our logic (The model output might be complex like "Acne and Rosacea")
    if "Acne" in disease_name: simplified_name = "Acne"
    elif "Eczema" in disease_name: simplified_name = "Eczema"
    elif "Psoriasis" in disease_name: simplified_name = "Psoriasis"
    else: simplified_name = disease_name # Fallback
    
    return simplified_name, confidence.item()

def generate_report(skin_type, disease_name, confidence_score):
    """Combines inputs to generate the final doctor-style report."""
    
    # Calculate Intensity Score (0-10) based on AI confidence
    # If AI is 90% sure, we treat it as high intensity/visibility
    intensity_score = round(confidence_score * 10, 1)
    
    # Get logic from Knowledge Base
    # Default to "General Advice" if the exact combination isn't found
    kb_entry = KNOWLEDGE_BASE.get(skin_type, {}).get(disease_name, 
               {"risk": "Unknown", "advice": "Consult a dermatologist for specific diagnosis.", "urgent": True})
    
    print("\n" + "="*40)
    print("     DERMATOLOGY AI ANALYSIS REPORT     ")
    print("="*40)
    print(f"1. PATIENT PROFILE")
    print(f"   - Detected Skin Type:  {skin_type}")
    print(f"   - Detected Condition:  {disease_name}")
    print("-" * 40)
    print(f"2. SEVERITY ANALYSIS")
    print(f"   - Intensity Score:     {intensity_score}/10")
    print(f"   - Risk Factor:         {kb_entry['risk'].upper()}")
    print("-" * 40)
    print(f"3. RECOMMENDATION")
    print(f"   - Advice: {kb_entry['advice']}")
    
    if kb_entry['urgent'] or intensity_score > 8.5:
        print("\n   [!] URGENT: PLEASE VISIT A DOCTOR.")
        print("       The intensity or combination suggests immediate professional care.")
    else:
        print("\n   [i] Note: Monitor daily. If symptoms worsen, visit a clinic.")
    print("="*40 + "\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load Models
    type_model, disease_model = load_models()
    
    # INPUTS: You can change these to input() to ask the user
    print("\n--- Step 1: Skin Type Analysis ---")
    img_path_1 = input("Enter path to Skin Type Image (or press Enter to reuse 'face.jpg'): ") or "face.jpg"
    detected_type = predict_skin_type(type_model, img_path_1)
    print(f"> Skin Type Detected: {detected_type}")
    
    print("\n--- Step 2: Disease Analysis ---")
    img_path_2 = input("Enter path to Disease Area Image (or press Enter to reuse same image): ") or img_path_1
    detected_disease, confidence = predict_disease(disease_model, img_path_2)
    print(f"> Disease Detected: {detected_disease} ({confidence*100:.1f}%)")
    
    # Step 3: Final Output
    generate_report(detected_type, detected_disease, confidence)