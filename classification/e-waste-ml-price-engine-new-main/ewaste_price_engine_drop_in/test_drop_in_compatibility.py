from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parent
PE = ROOT / "price_engine"
sys.path.insert(0, str(PE))
sys.path.insert(0, str(ROOT))

import legacy_price_engine
from price_engine import PriceEngine, QuoteRequest
from price_integration import recommend_price

dataset = PE / "data" / "price_dataset.csv"

old = legacy_price_engine.PriceEngine(dataset)
new = PriceEngine(dataset)

cases = [
    QuoteRequest(
        category="PCB",
        subcategory="Desktop PCB - Single Chip",
        state="Uttar Pradesh",
        city="Lucknow",
        quantity=5,
        total_weight_kg=12.5,
        channel="authorized",
    ),
    QuoteRequest(
        category="LCD",
        state="Delhi",
        city="Delhi",
        quantity=2,
        total_weight_kg=18,
        channel="mandi",
    ),
    QuoteRequest(
        category="CRT",
        state="Maharashtra",
        city="Mumbai",
        quantity=3,
        total_weight_kg=0,
        channel="informal",
    ),
    QuoteRequest(
        category="wires",
        state="Uttar Pradesh",
        city="Ghaziabad",
        quantity=10,
        total_weight_kg=22,
        channel="authorized",
    ),
]

for i, req in enumerate(cases, 1):
    old_req = legacy_price_engine.QuoteRequest(**req.__dict__)
    old_out = old.quote(old_req)
    new_out = new.quote(req)

    assert list(old_out.keys()) == list(new_out.keys()), (
        f"Case {i}: output keys/order changed.\n"
        f"OLD={list(old_out.keys())}\nNEW={list(new_out.keys())}"
    )

    assert new_out["unit"] == old_out["unit"]
    assert new_out["match_level"] == old_out["match_level"]
    assert new_out["calculation_basis"] == old_out["calculation_basis"]

    assert new_out["informal_rate_inr"] <= new_out["mandi_rate_inr"] <= new_out["authorized_rate_inr"]
    assert new_out["market_rate_min_inr"] <= new_out["recommended_rate_inr"] <= new_out["market_rate_max_inr"]

    mult = req.total_weight_kg if new_out["unit"] == "per_kg" else req.quantity
    assert abs(new_out["estimated_value_inr"] - mult * new_out["recommended_rate_inr"]) <= 1.0

# Existing integration endpoint/function contract.
integrated = recommend_price(
    {"category": "PCB", "confidence": 0.91},
    {
        "state": "Uttar Pradesh",
        "city": "Lucknow",
        "quantity": 5,
        "total_weight_kg": 12.5,
        "subcategory": "Desktop PCB - Single Chip",
        "channel": "authorized",
        "unit": "auto",
    },
)
assert list(integrated.keys()) == ["classification", "pricing"]
assert integrated["classification"]["predicted_category"] == "PCB"
assert integrated["classification"]["confidence"] == 0.91

# Per-piece predictions must stay positive; older XGBoost versions silently
# returned zero for the serialized per-piece models.
piece_quote = new.quote(QuoteRequest(
    category="MOBILE_TABLETS", state="Uttar Pradesh", city="Lucknow",
    quantity=2, total_weight_kg=0,
))
assert piece_quote["unit"] == "per_piece"
assert piece_quote["recommended_rate_inr"] > 0

# A city seen in another state must not be labeled an exact location match.
cross_state = new._ml.quote(
    category="MOBILE_TABLETS", state="Maharashtra", city="Lucknow",
    quantity=2, total_weight_kg=0,
)
assert cross_state["match_level"] == "state"

# This exact material/city previously crossed: mandi exceeded authorized.
ordered_piece = new._ml.quote(
    category="MOBILE_TABLETS", subcategory="Keypad Phone - Complete",
    state="Delhi NCR", city="Delhi", quantity=1, total_weight_kg=0,
)
assert (ordered_piece["informal_rate_inr"] <= ordered_piece["mandi_rate_inr"]
        <= ordered_piece["authorized_rate_inr"])

# The displayed total should agree with the displayed rate, even for large lots.
large_quote = new.quote(QuoteRequest(
    category="PCB", state="Uttar Pradesh", city="Lucknow",
    quantity=1, total_weight_kg=10000,
))
assert math.isclose(large_quote["estimated_value_inr"],
                    round(10000 * large_quote["recommended_rate_inr"], 2), abs_tol=0.01)

for invalid in (float("nan"), float("inf"), -float("inf")):
    try:
        new.quote(QuoteRequest(
            category="PCB", state="Uttar Pradesh", city="Lucknow",
            quantity=1, total_weight_kg=invalid,
        ))
    except ValueError:
        pass
    else:
        raise AssertionError(f"Non-finite weight accepted: {invalid}")

# Keys used by the existing camera_scan.py UI.
for key in [
    "recommended_rate_inr",
    "estimated_value_inr",
    "estimated_value_min_inr",
    "estimated_value_max_inr",
    "unit",
    "match_level",
]:
    assert key in integrated["pricing"]

print("DROP-IN COMPATIBILITY TESTS PASSED")
print("Same QuoteRequest inputs: YES")
print("Same PriceEngine constructor/quote endpoint: YES")
print("Same output key schema/order: YES")
print("Same recommend_price(classifier_result, user_input) endpoint: YES")
print("Same camera_scan.py-required pricing keys: YES")
print("Channel invariant informal <= mandi <= authorized: YES")
