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
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# MODEL SETTINGS
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

model = tf.keras.models.load_model(MODEL_PATH)

print("ML model loaded successfully.")


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(image_bytes):

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Could not decode image.")

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # Prediction
    predictions = model.predict(
        image,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_index]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return predicted_class, confidence


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "message": "E-waste ML API is running",
        "status": "running"
    })


# ============================================================
# IMAGE CLASSIFICATION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400

        image_file = request.files["image"]

        image_bytes = image_file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "Uploaded image is empty."
            }), 400

        # Predict
        category, confidence = predict_image(
            image_bytes
        )

        # Confidence status
        accepted = (
            confidence >= CONFIDENCE_THRESHOLD
        )

        return jsonify({

            "success": True,

            "classification": {

                "predicted_category": category,

                "confidence": confidence,

                "confidence_percent": round(
                    confidence * 100,
                    2
                ),

                "accepted": accepted

            }

        })

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# PRICE RECOMMENDATION API
# ============================================================

@app.route("/api/recommend-price", methods=["POST"])
def recommend():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400

        # ----------------------------------------------------
        # Required information
        # ----------------------------------------------------

        category = data["category"]

        confidence = float(
            data.get(
                "confidence",
                0
            )
        )

        state = data["state"]

        city = data["city"]

        quantity = float(
            data["quantity"]
        )

        total_weight = float(
            data["total_weight_kg"]
        )

        subcategory = data.get(
            "subcategory"
        )

        # ----------------------------------------------------
        # Prepare classifier result
        # ----------------------------------------------------

        classifier_result = {

            "category": category,

            "confidence": confidence

        }

        # ----------------------------------------------------
        # Prepare user input
        # ----------------------------------------------------

        user_input = {

            "state": state,

            "city": city,

            "quantity": quantity,

            "total_weight_kg": total_weight,

            "subcategory": subcategory

        }

        # ----------------------------------------------------
        # Get price recommendation
        # ----------------------------------------------------

        result = recommend_price(

            classifier_result,

            user_input

        )

        return jsonify({

            "success": True,

            "result": result

        })

    except KeyError as e:

        return jsonify({

            "success": False,

            "error": f"Missing required field: {str(e)}"

        }), 400

    except Exception as e:

        print(
            "Pricing error:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print(
        "      E-WASTE ML CLASSIFICATION & PRICING API"
    )

    print("=" * 60)

    print()

    print(
        "Health API:"
    )

    print(
        "http://127.0.0.1:5000/api/health"
    )

    print()

    print(
        "Classification API:"
    )

    print(
        "POST http://127.0.0.1:5000/api/predict"
    )

    print()

    print(
        "Pricing API:"
    )

    print(
        "POST http://127.0.0.1:5000/api/recommend-price"
    )

    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )