"""Drop-in ML replacement for the original e-waste PriceEngine.

PUBLIC CONTRACT IS PRESERVED:
- QuoteRequest fields are unchanged.
- PriceEngine(dataset_path) constructor is unchanged.
- PriceEngine.quote(QuoteRequest) is unchanged.
- Returned dictionary uses the same output keys as the original engine.
- CLI arguments are unchanged.
- Existing price_integration.py can import this module without frontend/backend changes.

Internally, the old deterministic engine is used only to preserve validation,
location fallback, metadata/provenance, and the response contract. The actual
three pricing rates are produced by the trained ML regressors.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from legacy_price_engine import PriceEngine as _LegacyPriceEngine
from ml_price_engine import MLPriceEngine


@dataclass(frozen=True)
class QuoteRequest:
    category: str
    state: str
    city: str
    quantity: float
    total_weight_kg: float
    channel: str = "authorized"
    unit: str = "auto"
    subcategory: Optional[str] = None
    as_of_date: Optional[str] = None


class PriceEngine:
    """Backward-compatible public engine powered by ML regression."""

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self._legacy = _LegacyPriceEngine(self.dataset_path)
        self._ml = MLPriceEngine()

        # Residual distributions from held-out-city validation.
        residual_path = Path(__file__).resolve().parent / "holdout_predictions.csv"
        self._residuals = pd.read_csv(residual_path) if residual_path.exists() else pd.DataFrame()

        # Avoid process-spawning overhead in API requests while keeping identical predictions.
        for unit_models in self._ml.models.values():
            for pipe in unit_models.values():
                model = pipe.named_steps.get("model")
                if hasattr(model, "n_jobs"):
                    model.n_jobs = 1

    @classmethod
    def canonical_category(cls, category: str) -> str:
        return _LegacyPriceEngine.canonical_category(category)

    @staticmethod
    def _ordered_channels(informal: float, mandi: float, authorized: float):
        """Nearest non-decreasing I<=M<=A rates via isotonic projection."""
        y = np.asarray([informal, mandi, authorized], dtype=float)
        ordered = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
            np.arange(3, dtype=float), y
        )
        return tuple(float(max(0.0, x)) for x in ordered)

    def _residual_quantiles(self, unit: str, channel: str):
        if self._residuals.empty:
            return (0.0, 0.0, 0.0, 0.0)
        d = self._residuals[
            (self._residuals["unit"] == unit) &
            (self._residuals["channel"] == channel)
        ].copy()
        if d.empty:
            return (0.0, 0.0, 0.0, 0.0)
        r = d["actual_rate_inr"].astype(float) - d["predicted_rate_inr"].astype(float)
        return tuple(float(r.quantile(q)) for q in (0.10, 0.25, 0.75, 0.90))

    @staticmethod
    def _spread_level(relative_iqr: float) -> str:
        if relative_iqr <= 0.25:
            return "low"
        if relative_iqr <= 0.75:
            return "medium"
        return "high"

    def _matched_location_pairs(self, legacy_quote: Dict[str, object]):
        # Preserve old location fallback behavior, but evaluate ML at the actual
        # matched city/state pairs rather than inventing a new HTTP/API contract.
        states = set(str(x) for x in legacy_quote.get("matched_states", []))
        cities = set(str(x) for x in legacy_quote.get("matched_cities", []))
        df = self._legacy.df

        m = df[
            df["location_state"].astype(str).isin(states) &
            df["location_city"].astype(str).isin(cities)
        ][["location_state", "location_city"]].drop_duplicates()

        pairs = [(str(r.location_state), str(r.location_city)) for r in m.itertuples(index=False)]
        if not pairs:
            pairs = [(
                str(legacy_quote["requested_state"]),
                str(legacy_quote["requested_city"]),
            )]
        return pairs

    def quote(self, request: QuoteRequest) -> Dict[str, object]:
        for name in ("quantity", "total_weight_kg"):
            value = getattr(request, name)
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        # Step 1: run the old contract/validation layer. This preserves:
        # aliases, supported inputs, unit rules, location fallback, metadata,
        # provenance fields, warnings structure, and all response keys.
        legacy_request = __import__("legacy_price_engine").QuoteRequest(
            category=request.category,
            state=request.state,
            city=request.city,
            quantity=request.quantity,
            total_weight_kg=request.total_weight_kg,
            channel=request.channel,
            unit=request.unit,
            subcategory=request.subcategory,
            as_of_date=request.as_of_date,
        )
        out = self._legacy.quote(legacy_request)

        category = str(out["category"])
        matched_subcategory = out["matched_subcategory"] if out["subcategory_matched"] else None
        unit = str(out["unit"])
        channel = str(out["pricing_channel"])

        # Step 2: ML rates. For fallback quotes, predict at all matched old-engine
        # locations and use the median, preserving the old location semantics.
        predictions = []
        for state, city in self._matched_location_pairs(out):
            q = self._ml.quote(
                category=category,
                state=state,
                city=city,
                quantity=float(request.quantity),
                total_weight_kg=float(request.total_weight_kg),
                subcategory=matched_subcategory,
                channel=channel,
                as_of_date=request.as_of_date or out["pricing_date"],
            )
            predictions.append(q)

        informal = float(np.median([q["informal_rate_inr"] for q in predictions]))
        mandi = float(np.median([q["mandi_rate_inr"] for q in predictions]))
        authorized = float(np.median([q["authorized_rate_inr"] for q in predictions]))

        # Enforce the business invariant found during stress testing.
        informal, mandi, authorized = self._ordered_channels(informal, mandi, authorized)

        rate_by_channel = {
            "informal": informal,
            "mandi": mandi,
            "authorized": authorized,
        }
        recommended = rate_by_channel[channel]

        # Step 3: empirical uncertainty using held-out-city residuals.
        rq10, rq25, rq75, rq90 = self._residual_quantiles(unit, channel)
        q10 = max(0.0, recommended + rq10)
        q25 = max(0.0, recommended + rq25)
        q75 = max(q25, recommended + rq75)
        q90 = max(q75, recommended + rq90)

        # Guarantee displayed interval contains the point estimate.
        min_rate = min(q10, recommended)
        max_rate = max(q90, recommended)
        iqr = max(0.0, q75 - q25)
        relative_iqr = float(iqr / recommended) if recommended > 0 else 0.0
        spread_level = self._spread_level(relative_iqr)

        multiplier = (
            float(request.total_weight_kg)
            if unit == "per_kg"
            else float(request.quantity)
        )

        # Step 4: overwrite ONLY values, never the public response schema.
        out["informal_rate_inr"] = round(informal, 2)
        out["mandi_rate_inr"] = round(mandi, 2)
        out["authorized_rate_inr"] = round(authorized, 2)
        out["recommended_rate_inr"] = round(recommended, 2)

        out["rate_q10_inr"] = round(q10, 2)
        out["rate_q25_inr"] = round(q25, 2)
        out["rate_q75_inr"] = round(q75, 2)
        out["rate_q90_inr"] = round(q90, 2)
        out["rate_iqr_inr"] = round(iqr, 2)
        out["relative_iqr"] = round(relative_iqr, 3)
        out["spread_level"] = spread_level
        out["market_rate_min_inr"] = round(min_rate, 2)
        out["market_rate_max_inr"] = round(max_rate, 2)
        out["range_method"] = "ml_holdout_residual_quantiles"

        out["estimated_value_inr"] = round(multiplier * out["recommended_rate_inr"], 2)
        out["estimated_value_min_inr"] = round(multiplier * out["market_rate_min_inr"], 2)
        out["estimated_value_max_inr"] = round(multiplier * out["market_rate_max_inr"], 2)

        return out


def main() -> None:
    # Same CLI endpoint/arguments as the old engine.
    parser = argparse.ArgumentParser(description="Calculate an e-waste scrap price quote")
    parser.add_argument("--dataset", default=Path(__file__).resolve().parent / "data" / "price_dataset.csv")
    parser.add_argument("--category", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--quantity", required=True, type=float)
    parser.add_argument("--weight", required=True, type=float, help="Total weight in kg")
    parser.add_argument("--channel", choices=["authorized", "informal", "mandi"], default="authorized")
    parser.add_argument("--unit", choices=["auto", "per_kg", "per_piece"], default="auto")
    parser.add_argument("--subcategory", default=None)
    parser.add_argument("--as-of-date", default=None, help="Optional quote date, e.g. 2026-09-05")
    args = parser.parse_args()

    engine = PriceEngine(args.dataset)
    result = engine.quote(
        QuoteRequest(
            category=args.category,
            state=args.state,
            city=args.city,
            quantity=args.quantity,
            total_weight_kg=args.weight,
            channel=args.channel,
            unit=args.unit,
            subcategory=args.subcategory,
            as_of_date=args.as_of_date,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
