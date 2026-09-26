
from __future__ import annotations

from pathlib import Path
from difflib import get_close_matches
from datetime import date
import json
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"

with open(BASE / "model_metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

CATEGORY_ALIASES = {
    "CRT": "CRT",
    "LCD": "LCD_LED",
    "LED": "LCD_LED",
    "LCD_LED": "LCD_LED",
    "LCD/LED": "LCD_LED",
    "LCD/LED DISPLAYS": "LCD_LED",
    "PCB": "PCB",
    "CABLE": "CABLES_WIRES",
    "CABLES": "CABLES_WIRES",
    "WIRE": "CABLES_WIRES",
    "WIRES": "CABLES_WIRES",
    "CABLES_WIRES": "CABLES_WIRES",
    "CABLES & WIRES": "CABLES_WIRES",
    "BATTERY": "BATTERIES",
    "BATTERIES": "BATTERIES",
    "MOTOR": "MOTORS",
    "MOTORS": "MOTORS",
    "PLASTIC": "PLASTICS",
    "PLASTICS": "PLASTICS",
    "MOBILE": "MOBILE_TABLETS",
    "MOBILES": "MOBILE_TABLETS",
    "MOBILE_TABLETS": "MOBILE_TABLETS",
}

STATE_ALIASES = {
    "up": "Uttar Pradesh",
    "u.p.": "Uttar Pradesh",
    "delhi": "Delhi NCR",
    "nct delhi": "Delhi NCR",
    "maharastra": "Maharashtra",
}

CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
}

CHANNELS = {"informal", "mandi", "authorized"}

class MLPriceEngine:
    def __init__(self):
        self.meta = META
        self.models = {}
        for unit, um in self.meta["units"].items():
            self.models[unit] = {}
            for channel, cm in um["channels"].items():
                self.models[unit][channel] = joblib.load(MODELS_DIR / cm["model_file"])

        self.known_locations = {
            (str(x["location_state"]).strip().lower(), str(x["location_city"]).strip().lower())
            for x in self.meta["city_tiers"]
        }
        self.known_states = {
            str(x["location_state"]).strip().lower() for x in self.meta["city_tiers"]
        }

    @staticmethod
    def _norm(s):
        return str(s).strip()

    def canonical_category(self, category):
        key = self._norm(category).upper()
        if key in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[key]
        supported = self.meta["supported_categories"]
        matches = get_close_matches(key, supported, n=3, cutoff=0.5)
        raise ValueError(f"Unknown category {category!r}. Supported: {supported}. Suggestions: {matches}")

    def canonical_state(self, state):
        raw = self._norm(state)
        return STATE_ALIASES.get(raw.lower(), raw)

    def canonical_city(self, city):
        raw = self._norm(city)
        return CITY_ALIASES.get(raw.lower(), raw)

    def _city_tier(self, state, city):
        for x in self.meta["city_tiers"]:
            if (str(x["location_state"]).strip().lower() == state.lower()
                and str(x["location_city"]).strip().lower() == city.lower()):
                return x["city_tier"]
        return "UNKNOWN"

    def _match_level(self, state, city):
        if (state.lower(), city.lower()) in self.known_locations:
            return "city"
        if state.lower() in self.known_states:
            return "state"
        return "national"

    def _row(self, category, subcategory, state, city, when):
        dt = pd.Timestamp(when)
        return {
            "vision_category": category,
            "subcategory": subcategory,
            "location_state": state,
            "location_city": city,
            "city_tier": self._city_tier(state, city),
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "day_of_year": dt.dayofyear,
        }

    def _predict_channel(self, unit, channel, rows):
        X = pd.DataFrame(rows)
        p = self.models[unit][channel].predict(X)
        return np.clip(np.asarray(p, dtype=float), 0, None)

    def quote(
        self,
        *,
        category,
        state,
        city,
        quantity=0.0,
        total_weight_kg=0.0,
        subcategory=None,
        channel="authorized",
        as_of_date=None,
    ):
        category = self.canonical_category(category)
        state = self.canonical_state(state)
        city = self.canonical_city(city)
        channel = str(channel).strip().lower()

        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {sorted(CHANNELS)}")
        if not math.isfinite(float(quantity)) or not math.isfinite(float(total_weight_kg)):
            raise ValueError("quantity and total_weight_kg must be finite numbers")
        if float(quantity) < 0 or float(total_weight_kg) < 0:
            raise ValueError("quantity and total_weight_kg cannot be negative")

        unit = self.meta["category_unit_map"][category]
        valid_subcats = self.meta["subcategories"].get(category, [])

        if subcategory:
            if subcategory not in valid_subcats:
                suggestions = get_close_matches(str(subcategory), valid_subcats, n=5, cutoff=0.45)
                raise ValueError(
                    f"Unknown subcategory {subcategory!r} for {category}. "
                    f"Suggestions: {suggestions}"
                )
            subcats_to_score = [subcategory]
            subcategory_mode = "exact"
        else:
            # Broad-category quote = median of model predictions across all known
            # subcategories in that category.
            subcats_to_score = valid_subcats
            subcategory_mode = "broad_median"

        when = as_of_date or self.meta["max_date_recorded"]
        rows = [self._row(category, sc, state, city, when) for sc in subcats_to_score]

        predicted = {}
        for ch in ["informal", "mandi", "authorized"]:
            vals = self._predict_channel(unit, ch, rows)
            predicted[ch] = float(np.median(vals))

        # Independently trained models can cross (e.g. mandi > authorized).
        # Project the three rates to their nearest non-decreasing values so
        # direct MLPriceEngine users get the same channel guarantee as the wrapper.
        channels = ["informal", "mandi", "authorized"]
        ordered = IsotonicRegression(increasing=True).fit_transform(
            np.arange(3), [predicted[ch] for ch in channels]
        )
        predicted.update(zip(channels, (float(x) for x in ordered)))

        recommended = predicted[channel]
        cm = self.meta["units"][unit]["channels"][channel]
        low = max(0.0, recommended + float(cm["residual_q10"]))
        high = max(low, recommended + float(cm["residual_q90"]))

        if unit == "per_kg":
            multiplier = float(total_weight_kg)
            multiplier_label = "total_weight_kg"
            if multiplier <= 0:
                raise ValueError("total_weight_kg must be > 0 for per_kg categories")
        else:
            multiplier = float(quantity)
            multiplier_label = "quantity"
            if multiplier <= 0:
                raise ValueError("quantity must be > 0 for per_piece categories")

        warnings = []
        if self.meta.get("is_synthetic_counts", {}).get("True", 0):
            warnings.append(
                "This hybrid dataset has source-backed material baselines, but its training "
                "price rows are synthetically expanded scenarios. Validate against independently "
                "observed transactions before claiming real-market predictive accuracy."
            )
        match_level = self._match_level(state, city)
        if match_level != "city":
            warnings.append(
                f"Exact city was not seen in model training; {match_level}-level generalization is being used."
            )
        if subcategory_mode == "broad_median":
            warnings.append(
                "No subcategory was supplied; the quote is the median across known subcategory predictions."
            )

        return {
            "category": category,
            "subcategory": subcategory,
            "subcategory_mode": subcategory_mode,
            "state": state,
            "city": city,
            "match_level": match_level,
            "as_of_date": str(pd.Timestamp(when).date()),
            "unit": unit,
            "model_used": cm["best_model"],
            "informal_rate_inr": round(predicted["informal"], 2),
            "mandi_rate_inr": round(predicted["mandi"], 2),
            "authorized_rate_inr": round(predicted["authorized"], 2),
            "pricing_channel": channel,
            "recommended_rate_inr": round(recommended, 2),
            "rate_interval_low_inr": round(low, 2),
            "rate_interval_high_inr": round(high, 2),
            "validation_mae_inr": round(float(cm["holdout_mae_inr"]), 2),
            "calculation_basis": f"{multiplier_label} × recommended_rate_inr",
            "estimated_value_inr": round(multiplier * recommended, 2),
            "estimated_value_low_inr": round(multiplier * low, 2),
            "estimated_value_high_inr": round(multiplier * high, 2),
            "warnings": warnings,
        }

if __name__ == "__main__":
    engine = MLPriceEngine()
    demo = engine.quote(
        category="PCB",
        subcategory="Desktop PCB - Single Chip",
        state="Uttar Pradesh",
        city="Lucknow",
        quantity=5,
        total_weight_kg=12.5,
        channel="authorized",
    )
    print(json.dumps(demo, indent=2, ensure_ascii=False))
