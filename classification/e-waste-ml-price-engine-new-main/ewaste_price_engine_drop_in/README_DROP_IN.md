# Drop-in ML Price Engine Upgrade

This package is designed to replace the old pricing engine without changing the
frontend/camera integration contract.

## Preserved inputs

`QuoteRequest` still accepts:

- category
- state
- city
- quantity
- total_weight_kg
- channel
- unit
- subcategory
- as_of_date

## Preserved public endpoints

### Python engine
```python
engine = PriceEngine(DATASET_PATH)
quote = engine.quote(QuoteRequest(...))
```

### Existing integration function
```python
recommend_price(classifier_result, user_input)
```

The outer integration response is still:

```python
{
    "classification": {...},
    "pricing": {...}
}
```

### CLI
`price_engine.py` keeps the same command-line arguments:
`--dataset`, `--category`, `--state`, `--city`, `--quantity`, `--weight`,
`--channel`, `--unit`, `--subcategory`, `--as-of-date`.

## Preserved outputs

The ML adapter intentionally returns the same response keys, in the same order,
as the previous deterministic `PriceEngine.quote()` response. Existing UI fields
such as `recommended_rate_inr`, `estimated_value_inr`,
`estimated_value_min_inr`, `estimated_value_max_inr`, `unit`, and `match_level`
remain unchanged.

## Internal change only

The pricing rates are now produced by the trained regression models. The old
engine is retained internally only as a compatibility/metadata layer for input
validation, category aliases, location fallback, unit resolution, provenance,
and response schema.

A post-prediction monotonic constraint guarantees:

`informal_rate_inr <= mandi_rate_inr <= authorized_rate_inr`

without changing the public API.
