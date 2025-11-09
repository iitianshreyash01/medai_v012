import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import logging

# ======================
# INITIAL SETUP
# ======================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ Correct CORS configuration
CORS(app, resources={r"/api/*": {
    "origins": [
        "https://iitianshreyash01.github.io",  # GitHub Pages
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}}, supports_credentials=True)

# ======================
# GEMINI MODEL SETUP
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured successfully")

model = None
try:
    for model_name in ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]:
        try:
            logger.info(f"Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            logger.info(f"✅ Successfully initialized model: {model_name}")
            break
        except Exception as e:
            logger.warning(f"Model {model_name} not available: {str(e)[:100]}")
    if not model:
        logger.error("❌ No available models found.")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    model = None

# ======================
# ROUTES
# ======================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check for Render + Gemini"""
    model_name = "Unknown"
    if model:
        try:
            model_name = getattr(model, "model_name", "Initialized")
        except Exception:
            model_name = "Initialized"
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "gemini_configured": GEMINI_API_KEY is not None,
        "model": model_name
    }), 200


@app.route("/api/ai-doctor", methods=["POST", "OPTIONS"])
def ai_doctor():
    """AI health assistant powered by Gemini"""

    # ✅ Handle preflight (CORS) requests
    if request.method == "OPTIONS":
        response = jsonify({"status": "CORS preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "https://iitianshreyash01.github.io")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response, 200

    # ✅ Explicit CORS headers for POST response
    response_headers = {
        "Access-Control-Allow-Origin": "https://iitianshreyash01.github.io",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
    }

    # ✅ Define system prompt
    system_prompt = """You are MedAI Pro, an AI health assistant. Provide CONCISE, PRACTICAL advice.

Response Format:
1. **Symptoms Analysis**: 2–3 lines about likely cause.
2. **Possible Causes**: 2–3 bullet points.
3. **Home Care**: 2–3 remedies.
4. **Medicines**: Include examples (Paracetamol, Ibuprofen, etc.).
5. **When to See Doctor**: When to seek professional help.

⚠️ Keep total response <150 words. Always end with: “⚠️ This is NOT professional medical advice.”"""

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400

        if not model:
            return jsonify({"error": "AI model not initialized.", "success": False}), 500

        full_message = f"{system_prompt}\n\nUser Symptom: {user_input}"

        logger.info(f"Processing query: {user_input[:60]}...")

        response = model.generate_content(
            full_message,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.7,
            )
        )

        if response and getattr(response, "text", None):
            result = jsonify({
                "success": True,
                "response": response.text,
                "timestamp": datetime.now().isoformat()
            })
        else:
            result = jsonify({
                "success": False,
                "error": "Empty response from AI model."
            })
        for k, v in response_headers.items():
            result.headers.add(k, v)
        return result, 200

    except Exception as e:
        logger.error(f"Error in ai_doctor endpoint: {str(e)}")
        result = jsonify({
            "error": f"Server error: {str(e)[:100]}",
            "success": False
        })
        for k, v in response_headers.items():
            result.headers.add(k, v)
        return result, 500


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


@app.route("/api/health-tips", methods=["GET"])
def get_health_tips():
    tips = [
        "Stay hydrated - at least 8 glasses of water daily.",
        "Exercise regularly for 30 minutes a day.",
        "Sleep 7–9 hours every night.",
        "Eat fruits and vegetables daily.",
        "Practice stress management (yoga, breathing).",
    ]
    return jsonify({"success": True, "tips": tips}), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
