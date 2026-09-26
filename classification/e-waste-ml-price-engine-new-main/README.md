# E-waste ML price engine

Drop-in pricing component for an e-waste image classification application. The trained regressors, datasets, compatibility wrapper, integration function, and test are in [`ewaste_price_engine_drop_in/`](ewaste_price_engine_drop_in/).

## Run locally

Use Python 3.12 (the tested version). From the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r ewaste_price_engine_drop_in/price_engine/requirements.txt
python ewaste_price_engine_drop_in/restore_models.py
python ewaste_price_engine_drop_in/test_drop_in_compatibility.py
```

The restore command checks the bundle SHA-256 and extracts six model files into `price_engine/models/`; it is needed once per fresh clone. The test should print `DROP-IN COMPATIBILITY TESTS PASSED`. Keep the restored `models/`, `data/`, `model_metadata.json`, and `holdout_predictions.csv` files alongside the Python modules. Model files use `joblib`; load them only from this trusted repository. The requirements pin the versions used to build/test these models: incompatible XGBoost versions can load a model but produce incorrect per-piece quotes. Use Python 3.12 for the tested setup.

## Connect the image classifier

From the repository root, the classifier's label must be converted to a supported pricing category. The engine does **not** classify the image. Supply the location and measured item count/weight separately:

```python
from ewaste_price_engine_drop_in.price_integration import recommend_price

classifier_result = {"category": "PCB", "confidence": 0.91}
user_input = {
    "state": "Uttar Pradesh",
    "city": "Lucknow",
    "quantity": 5,
    "total_weight_kg": 12.5,
    "subcategory": "Desktop PCB - Single Chip",  # optional; omit if unknown
    "channel": "authorized",              # authorized, mandi, informal
    "unit": "auto",                       # auto, per_kg, per_piece
}
result = recommend_price(classifier_result, user_input)
print(result["pricing"]["recommended_rate_inr"])
print(result["pricing"]["estimated_value_inr"])
```

The classifier response needs a `category` field and may include `confidence`. Pricing accepts category aliases such as `LCD`, `LED`, `CRT`, `PCB`, `wires`, `battery`, `motor`, `plastic`, and `mobile`. Canonical categories are `LCD_LED`, `CRT`, `PCB`, `CABLES_WIRES`, `BATTERIES`, `MOTORS`, `PLASTICS`, and `MOBILE_TABLETS`. Map any different image model labels to one of these before calling the function. `subcategory` must be a recognized exact value if supplied; otherwise omit it for a broad-category quote. `quantity` and `total_weight_kg` are required numeric inputs. The result has `classification` and `pricing` objects; useful pricing keys include `recommended_rate_inr`, `estimated_value_inr`, `estimated_value_min_inr`, `estimated_value_max_inr`, `unit`, and `match_level`.

For an existing application that imports `price_integration.py` directly, run the restore command first, then copy `price_integration.py` and the `price_engine/` folder with its restored `models/` into the corresponding backend directory. The function signature and quote output fields match the supplied legacy engine; see [the compatibility notes](ewaste_price_engine_drop_in/README_DROP_IN.md). For a web API, call `recommend_price` in your backend after classification and return its dictionary as JSON. Load the engine once per process as the integration module does, rather than reloading the models for every request.

## Data and evaluation limits

The pricing data is **hybrid in provenance**: 65 curated material baselines carry source URLs, observation dates, and evidence levels (34 `DIRECT`, 24 `DIRECT_RANGE`, 7 `PROXY`). The 1,116 location/date/channel price rows were expanded from those baselines and are all flagged `is_synthetic=True`; they are scenario estimates, not 1,116 independent market observations. Held-out-city scores therefore measure performance on expanded scenarios, not verified real-market accuracy. The stress-test report documents 36 failed checks out of 6,194 on the underlying model layer; the drop-in wrapper applies channel ordering, and its supplied compatibility test passes. Validate quotes against independent local transactions before production use, especially across dates or unseen locations. See [model audit](ewaste_price_engine_drop_in/price_engine/ML_MODEL_AUDIT.md) and [stress test](ewaste_price_engine_drop_in/price_engine/STRESS_TEST_ML_REPORT.md).
