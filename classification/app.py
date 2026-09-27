import os
import json
import math
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import tensorflow as tf

from price_integration import recommend_price, engine

# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
CORS(app, origins=os.environ.get("CORS_ORIGINS", "*").split(","))

@app.errorhandler(413)
def too_large(error):
    return jsonify(success=False, error="Maximum upload size is 10 MiB."), 413

# ============================================================
# MODEL CONFIGURATION & CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_e_waste_model_v5.keras"
IMAGE_SIZE = (256, 256)
CLASS_NAMES = json.loads((BASE_DIR / "class_names.json").read_text())

CONFIDENCE_THRESHOLD = 0.70

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ML model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("Model output count does not match class_names.json")
    print("ML model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load classifier: {MODEL_PATH}") from e

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

    # Match the v5 camera_scan.py preprocessing: RGB float32 in [0, 255].
    image = image.astype("float32")

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
        "pricing_loaded": engine is not None,
        "model_version": "v5",
        "classes": CLASS_NAMES,
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
            "error": "Internal inference error."
        }), 500

# ============================================================
# PRICE RECOMMENDATION ENDPOINT
# ============================================================

@app.route("/api/recommend-price", methods=["POST"])
def recommend():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or not data:
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

        for field in ("category", "state", "city"):
            if not isinstance(data[field], str) or not data[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        for field in ("quantity", "total_weight_kg"):
            value = float(data[field])
            if isinstance(data[field], bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be a finite positive number")
        confidence = float(data.get("confidence", 0.0))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

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
            "subcategory": data.get("subcategory"),
            "channel": data.get("channel", "authorized"),
            "unit": data.get("unit", "auto"),
            "as_of_date": data.get("as_of_date")
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
            "error": "Internal pricing error."
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
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=False
    )
