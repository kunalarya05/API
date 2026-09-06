import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import tensorflow as tf

from price_integration import recommend_price

# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__)
CORS(app)  # Enables frontend/backend clients to connect across different ports/domains

# ============================================================
# MODEL CONFIGURATION & CONSTANTS
# ============================================================

MODEL_PATH = "best_e_waste_model.keras"
IMAGE_SIZE = (256, 256)

CLASS_NAMES = [
    "Battery",
    "CRT",
    "LCD_LED",
    "Motors",
    "PCB",
    "Plastic",
    "Wires"
]

CONFIDENCE_THRESHOLD = 0.70

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ML model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("ML model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error: Failed to load model from {MODEL_PATH}: {e}")

# ============================================================
# IMAGE PREPROCESSING & INFERENCE
# ============================================================

def predict_image(image_bytes):
    # Decode raw byte stream into OpenCV image array
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data. Unable to decode.")

    # Resize to the model's required input resolution
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert OpenCV standard BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel intensity to [0.0, 1.0]
    image = image.astype("float32") / 255.0

    # Add batch dimension: (256, 256, 3) -> (1, 256, 256, 3)
    image = np.expand_dims(image, axis=0)

    # Inference
    predictions = model.predict(image, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence

# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "message": "E-waste ML API is running",
        "model_loaded": model is not None,
        "status": "running"
    }), 200

# ============================================================
# IMAGE CLASSIFICATION ENDPOINT
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": "ML model is not loaded on server."
        }), 503

    try:
        # Check if form-data contains the 'image' field
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "Missing 'image' key in form-data payload."
            }), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({
                "success": False,
                "error": "Uploaded image file is empty."
            }), 400

        # Perform prediction
        category, confidence = predict_image(image_bytes)
        accepted = confidence >= CONFIDENCE_THRESHOLD

        return jsonify({
            "success": True,
            "classification": {
                "predicted_category": category,
                "confidence": round(confidence, 4),
                "confidence_percent": round(confidence * 100, 2),
                "accepted": accepted
            }
        }), 200

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({
            "success": False,
            "error": f"Internal inference error: {str(e)}"
        }), 500

# ============================================================
# PRICE RECOMMENDATION ENDPOINT
# ============================================================

@app.route("/api/recommend-price", methods=["POST"])
def recommend():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Missing or invalid JSON body in request."
            }), 400

        # Validate mandatory keys
        required_fields = ["category", "state", "city", "quantity", "total_weight_kg"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: '{field}'"
                }), 400

        # Payload assembly
        classifier_result = {
            "category": str(data["category"]),
            "confidence": float(data.get("confidence", 0.0))
        }

        user_input = {
            "state": str(data["state"]),
            "city": str(data["city"]),
            "quantity": float(data["quantity"]),
            "total_weight_kg": float(data["total_weight_kg"]),
            "subcategory": data.get("subcategory")
        }

        # Run price logic from imported module
        result = recommend_price(classifier_result, user_input)

        return jsonify({
            "success": True,
            "result": result
        }), 200

    except (ValueError, TypeError) as conv_err:
        return jsonify({
            "success": False,
            "error": f"Invalid data type provided: {str(conv_err)}"
        }), 400
    except Exception as e:
        print("Pricing calculation error:", e)
        return jsonify({
            "success": False,
            "error": f"Internal pricing error: {str(e)}"
        }), 500

# ============================================================
# SERVER STARTUP
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("      E-WASTE ML CLASSIFICATION & PRICING SERVER")
    print("=" * 60 + "\n")
    print("Endpoints:")
    print("  GET  /api/health")
    print("  POST /api/predict")
    print("  POST /api/recommend-price\n")

    # host='0.0.0.0' allows external connections from your LAN/teammate.
    # threaded=False ensures safe, single-thread TensorFlow tensor memory execution.
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=False
    )