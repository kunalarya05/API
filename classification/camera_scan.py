import cv2
import numpy as np
import tensorflow as tf
from price_integration import recommend_price

# SETTINGS

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



# LOAD MODEL


print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# PREDICTION FUNCTION

def predict_image(frame):
    """
    Takes one OpenCV camera frame and returns
    the predicted class and confidence.
    """

    # Resize image to model input size
    image = cv2.resize(frame, IMAGE_SIZE)

    # OpenCV uses BGR.
    # Convert BGR → RGB because training images were RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert pixel values 0-255 → 0-1
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # CNN prediction
    predictions = model.predict(image, verbose=0)

    # Find highest probability
    predicted_index = np.argmax(predictions[0])

    confidence = float(predictions[0][predicted_index])

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# CAMERA

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print()
print("=" * 50)
print("        E-WASTE LIVE SCANNER")
print("=" * 50)
print()
print("Press S → Scan object")
print("Press Q → Quit")
print()


# MAIN CAMERA LOOP

last_result = None

while True:

    # Capture frame
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break


    # Display instructions
    

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

    # Display previous scan result
    
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

            result_text = "Not confident - Try again"

            cv2.putText(
                frame,
                result_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    
    # Show camera
    

    cv2.imshow(
        "E-Waste Classification",
        frame
    )

    
    # Keyboard controls


    key = cv2.waitKey(1) & 0xFF

    # Press S → SCAN
    if key == ord("s"):

        print()
        print("Scanning...")

        predicted_class, confidence = predict_image(frame)

        last_result = (
            predicted_class,
            confidence
        )

        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")

        # Check confidence
        if confidence >= CONFIDENCE_THRESHOLD:

            print("Result: ACCEPTED")

            # Ask user to confirm prediction
            confirm = input(
                f"\nUse '{predicted_class}' for pricing? (y/n): "
            ).strip().lower()

            if confirm == "y":

                print("\nPrediction confirmed.")

                # Get user information
                state = input("Enter state: ").strip()
                city = input("Enter city: ").strip()

                quantity = float(
                    input("Enter quantity: ")
                )

                total_weight = float(
                    input("Enter total weight (kg): ")
                )

                # Prepare classifier result
                classifier_result = {
                    "category": predicted_class,
                    "confidence": confidence
                }

                # Prepare user input
                user_input = {
                    "state": state,
                    "city": city,
                    "quantity": quantity,
                    "total_weight_kg": total_weight
                }

                print("\nGetting price recommendation...")

                # Call price engine
                result = recommend_price(
                    classifier_result,
                    user_input
                )

                pricing = result["pricing"]

                # Display result
                print("\n" + "=" * 60)
                print("PRICE RECOMMENDATION")
                print("=" * 60)

                print(
                    f"Category          : {predicted_class}"
                )

                print(
                    f"Confidence        : {confidence * 100:.2f}%"
                )

                print(
                    f"Recommended Rate  : ₹{pricing['recommended_rate_inr']:.2f}"
                )

                print(
                    f"Estimated Value   : ₹{pricing['estimated_value_inr']:.2f}"
                )

                print(
                    f"Estimated Range   : "
                    f"₹{pricing['estimated_value_min_inr']:.2f}"
                    f" - "
                    f"₹{pricing['estimated_value_max_inr']:.2f}"
                )

                print(
                    f"Price Unit        : {pricing['unit']}"
                )

                print(
                    f"Match Level       : {pricing['match_level']}"
                )

                print("=" * 60)

            else:

                print(
                    "\nPrediction rejected. Press S to scan again."
                )

        else:

            print(
                "Result: LOW CONFIDENCE - TRY AGAIN"
            )

    # Press Q → QUIT
    elif key == ord("q"):

        break



# CLEANUP

camera.release()
cv2.destroyAllWindows()

print()
print("Camera closed.")