import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# PRICE INTEGRATION
# ============================================================

# Since price_integration.py is in the SAME folder,
# we can directly import it.
from price_integration import recommend_price


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_e_waste_model_v5.keras"

IMAGE_SIZE = (256, 256)

CLASS_NAMES = [
    "Battery",
    "CRT",
    "LCD_LED",
    "Mobile",
    "Motors",
    "PCB",
    "Plastic",
    "Wires"
]

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("       KABADIWALA CONNECT - E-WASTE SCANNER")
print("=" * 60)

print("\nLoading CNN model...")

if not MODEL_PATH.exists():
    print("\nERROR: Model file not found!")
    print(f"Expected location:")
    print(MODEL_PATH)
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CNN model loaded successfully.")

except Exception as e:
    print("\nERROR: Could not load CNN model.")
    print(e)
    exit()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(frame):
    """
    Takes an OpenCV camera frame and returns:
        predicted_class
        confidence
    """

    # Resize image
    image = cv2.resize(frame, IMAGE_SIZE)

    # OpenCV = BGR
    # TensorFlow training images = RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert 0-255 to 0-1
    image = image.astype("float32")

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # CNN prediction
    predictions = model.predict(image, verbose=0)

    # Highest probability
    predicted_index = int(np.argmax(predictions[0]))

    confidence = float(predictions[0][predicted_index])

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# ============================================================
# CAMERA
# ============================================================

print("\nOpening camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("\nERROR: Camera could not be opened.")
    print("Check whether another application is using your camera.")
    exit()

print("Camera opened successfully.")

print()
print("=" * 60)
print("CONTROLS")
print("=" * 60)
print("Press S  →  Scan object")
print("Press Q  →  Quit")
print("=" * 60)


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

last_result = None

while True:

    # Capture frame
    ret, frame = camera.read()

    if not ret:
        print("\nERROR: Could not read camera frame.")
        break

    # Keep the ORIGINAL frame for prediction.
    # This prevents the text drawn on the screen from
    # becoming part of the CNN input.
    original_frame = frame.copy()

    # --------------------------------------------------------
    # SCREEN TEXT
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Press S to SCAN",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to QUIT",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # DISPLAY LAST RESULT
    # --------------------------------------------------------

    if last_result is not None:

        label, confidence = last_result

        if confidence >= CONFIDENCE_THRESHOLD:

            result_text = f"{label}: {confidence * 100:.1f}%"

            cv2.putText(
                frame,
                result_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

        else:

            cv2.putText(
                frame,
                "Low confidence - Scan again",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # --------------------------------------------------------
    # SHOW CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "KabadIwala Connect - E-Waste Scanner",
        frame
    )

    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # SCAN
    # ========================================================

    if key == ord("s"):

        print()
        print("=" * 60)
        print("SCANNING...")
        print("=" * 60)

        try:
            predicted_class, confidence = predict_image(
                original_frame
            )

        except Exception as e:
            print("\nERROR during prediction:")
            print(e)
            continue

        # Save result
        last_result = (
            predicted_class,
            confidence
        )

        print(f"\nPrediction  : {predicted_class}")
        print(f"Confidence  : {confidence * 100:.2f}%")

        # ----------------------------------------------------
        # CONFIDENCE CHECK
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            print("\nResult: LOW CONFIDENCE")
            print("Please position the object clearly and scan again.")

            continue

        print("\nResult: ACCEPTED")

        # ----------------------------------------------------
        # CONFIRM PREDICTION
        # ----------------------------------------------------

        confirm = input(
            f"\nUse '{predicted_class}' for pricing? (y/n): "
        ).strip().lower()

        if confirm != "y":

            print("\nPrediction rejected.")
            print("Press S to scan again.")

            continue

        print("\nPrediction confirmed.")

        # ----------------------------------------------------
        # USER INFORMATION
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print("ENTER ITEM DETAILS")
        print("-" * 60)

        state = input(
            "Enter state: "
        ).strip()

        city = input(
            "Enter city: "
        ).strip()

        # Quantity
        while True:
            try:
                quantity = float(
                    input("Enter quantity: ")
                )

                if quantity <= 0:
                    print("Quantity must be greater than 0.")
                    continue

                break

            except ValueError:
                print("Please enter a valid number.")

        # Weight
        while True:
            try:
                total_weight = float(
                    input("Enter total weight (kg): ")
                )

                if total_weight <= 0:
                    print("Weight must be greater than 0.")
                    continue

                break

            except ValueError:
                print("Please enter a valid number.")

        # ----------------------------------------------------
        # CLASSIFIER RESULT
        # ----------------------------------------------------

        classifier_result = {
            "category": predicted_class,
            "confidence": confidence
        }

        # ----------------------------------------------------
        # USER INPUT FOR PRICE ENGINE
        # ----------------------------------------------------

        user_input = {
            "state": state,
            "city": city,
            "quantity": quantity,
            "total_weight_kg": total_weight,

            # New price engine defaults
            "channel": "authorized",
            "unit": "auto"
        }

        # ----------------------------------------------------
        # PRICE ENGINE
        # ----------------------------------------------------

        print()
        print("Getting price recommendation...")

        try:

            result = recommend_price(
                classifier_result,
                user_input
            )

        except Exception as e:

            print("\nERROR: Price engine failed.")
            print(e)

            print("\nPossible reasons:")
            print("1. State/city is not available in the pricing dataset.")
            print("2. Price engine dependencies are missing.")
            print("3. Price integration has a configuration issue.")

            continue

        # ----------------------------------------------------
        # GET PRICING
        # ----------------------------------------------------

        try:
            pricing = result["pricing"]

        except KeyError:

            print("\nERROR: Pricing result has unexpected format.")
            print(result)

            continue

        # ----------------------------------------------------
        # DISPLAY PRICE
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("             PRICE RECOMMENDATION")
        print("=" * 60)

        print(
            f"Category          : {predicted_class}"
        )

        print(
            f"Confidence        : {confidence * 100:.2f}%"
        )

        # Recommended rate
        if "recommended_rate_inr" in pricing:

            print(
                f"Recommended Rate  : ₹"
                f"{pricing['recommended_rate_inr']:.2f}"
            )

        # Estimated value
        if "estimated_value_inr" in pricing:

            print(
                f"Estimated Value   : ₹"
                f"{pricing['estimated_value_inr']:.2f}"
            )

        # Price range
        if (
            "estimated_value_min_inr" in pricing
            and
            "estimated_value_max_inr" in pricing
        ):

            print(
                f"Estimated Range   : ₹"
                f"{pricing['estimated_value_min_inr']:.2f}"
                f" - ₹"
                f"{pricing['estimated_value_max_inr']:.2f}"
            )

        # Unit
        if "unit" in pricing:

            print(
                f"Price Unit        : "
                f"{pricing['unit']}"
            )

        # Match level
        if "match_level" in pricing:

            print(
                f"Match Level       : "
                f"{pricing['match_level']}"
            )

        print("=" * 60)

        print("\nScan another object by pressing S.")
        print("Press Q to quit.")

    # ========================================================
    # QUIT
    # ========================================================

    elif key == ord("q"):

        print("\nClosing camera...")
        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print()
print("Camera closed.")
print("Thank you for using KabadIwala Connect!")