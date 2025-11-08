import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")

# Initialize Gemini model with smart fallback
model = None
try:
    models_to_try = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro"
    ]

    for model_name in models_to_try:
        try:
            logger.info(f"Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            logger.info(f"✅ Successfully initialized model: {model_name}")
            break
        except Exception as e:
            logger.warning(f"Model {model_name} not available: {str(e)[:100]}")
            continue

    if not model:
        logger.error("No available models found!")

except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}")
    model = None

# Health check endpoint
@app.route("/api/health", methods=["GET"])
def health_check():
    model_name = "Unknown"
    if model:
        try:
            model_name = model.model_name if hasattr(model, 'model_name') else str(model)
        except:
            model_name = "Initialized"

    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "gemini_configured": GEMINI_API_KEY is not None,
        "model": model_name
    }), 200

# AI Doctor endpoint - OPTIMIZED FOR CONCISE RESPONSES
@app.route("/api/ai-doctor", methods=["POST", "OPTIONS"])
def ai_doctor():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400

        if not model:
            return jsonify({
                "error": "AI model not initialized. Check your API key and try restarting.",
                "success": False
            }), 500

        # OPTIMIZED SYSTEM PROMPT - For precise, concise responses with medicine suggestions
        system_prompt = """You are MedAI Pro, an AI health assistant. Provide CONCISE, PRECISE health advice.

RESPONSE FORMAT (IMPORTANT - Follow exactly):
1. **Symptoms Analysis**: 2-3 sentences explaining what the symptoms indicate
2. **Possible Causes**: List 2-3 most likely causes (bullet points)
3. **Suggested Home Care**: 2-3 quick remedies
4. **Suggested Medicines**: 
   - Fever: Paracetamol/Ibuprofen
   - Cough: Dextromethorphan/Honey
   - Headache: Aspirin/Ibuprofen
   - Nausea: Ginger/Peppermint
   - Diarrhea: Loperamide/Electrolytes
   - Pain: Ibuprofen/Paracetamol
5. **When to See Doctor**: If symptoms persist >7 days or worsen

CRITICAL RULES:
- Be VERY BRIEF (max 150 words total)
- Use bullet points, not paragraphs
- Include specific medicine names and dosages if applicable
- NO lengthy explanations
- Focus on practical advice only
- Always add: ⚠️ This is NOT professional medical advice

Example response format:
**Symptoms Analysis**: You have gastroenteritis. Common viral infection causing vomiting.

**Likely Causes**:
- Viral gastroenteritis
- Food poisoning
- Bacterial infection

**Home Care**:
- Rest and stay hydrated
- Eat light foods (rice, toast)
- Ginger tea helps

**Medicines**:
- Metoclopramide 10mg (anti-nausea) - 3 times daily
- Electrolyte solutions for hydration
- Ibuprofen 400mg if fever present

**When to See Doctor**:
- Vomiting >24 hours
- Blood in vomit
- Severe abdominal pain

⚠️ This is NOT professional medical advice. Consult a doctor for diagnosis."""

        # Create message with system context
        full_message = f"{system_prompt}\n\nUser Symptom: {user_input}\n\nProvide response in the exact format above. Be VERY BRIEF and CONCISE."

        logger.info(f"Processing query: {user_input[:50]}...")

        # Generate response with constraints
        try:
            response = model.generate_content(
                full_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,  # Limit response length
                    temperature=0.7,  # More focused responses
                )
            )

            if response and response.text:
                logger.info("✅ Successfully generated response from Gemini")
                return jsonify({
                    "success": True,
                    "response": response.text,
                    "timestamp": datetime.now().isoformat()
                }), 200
            else:
                logger.error("Model returned empty response")
                return jsonify({
                    "error": "AI model returned empty response. Try again.",
                    "success": False
                }), 500

        except Exception as gemini_error:
            error_msg = str(gemini_error)
            logger.error(f"Gemini API error: {error_msg}")

            return jsonify({
                "error": f"API Error: {error_msg[:100]}",
                "success": False
            }), 500

    except Exception as e:
        logger.error(f"Error in ai_doctor endpoint: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)[:100]}",
            "success": False
        }), 500

# Specialists endpoint
@app.route("/api/specialists", methods=["GET"])
def get_specialists():
    specialists = [
        {"id": 1, "name": "Dr. Rajesh Kumar", "specialty": "Cardiology", "rating": 4.8},
        {"id": 2, "name": "Dr. Priya Sharma", "specialty": "Dermatology", "rating": 4.7},
        {"id": 3, "name": "Dr. Amit Patel", "specialty": "Neurology", "rating": 4.9},
        {"id": 4, "name": "Dr. Anjali Singh", "specialty": "Pediatrics", "rating": 4.6},
        {"id": 5, "name": "Dr. Vikram Gupta", "specialty": "Orthopedics", "rating": 4.8},
    ]
    return jsonify({"success": True, "specialists": specialists}), 200

# Health tips endpoint
@app.route("/api/health-tips", methods=["GET"])
def get_health_tips():
    tips = [
        "Stay hydrated - drink at least 8 glasses of water daily",
        "Exercise regularly - aim for 30 minutes of physical activity",
        "Get adequate sleep - 7-9 hours per night",
        "Eat balanced meals with fruits and vegetables",
        "Manage stress through meditation or yoga",
    ]
    return jsonify({"success": True, "tips": tips}), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
