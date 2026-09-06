from pathlib import Path
import sys

# Path to the price_engine folder
PRICE_ENGINE_DIR = Path(__file__).resolve().parent / "price_engine"

# Allow Python to find price_engine.py
sys.path.insert(0, str(PRICE_ENGINE_DIR))

from price_engine import PriceEngine, QuoteRequest


# Path to pricing dataset
DATASET_PATH = PRICE_ENGINE_DIR / "data" / "price_dataset.csv"

# Create pricing engine
engine = PriceEngine(DATASET_PATH)


def recommend_price(classifier_result, user_input):

    category = classifier_result["category"]

    quote = engine.quote(
        QuoteRequest(
            category=category,
            state=user_input["state"],
            city=user_input["city"],
            quantity=float(user_input["quantity"]),
            total_weight_kg=float(user_input["total_weight_kg"]),
            subcategory=user_input.get("subcategory"),
            channel=user_input.get("channel", "authorized"),
            unit=user_input.get("unit", "auto"),
            as_of_date=user_input.get("as_of_date"),
        )
    )

    return {
        "classification": {
            "predicted_category": category,
            "confidence": classifier_result.get("confidence"),
        },
        "pricing": quote,
    }