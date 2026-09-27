# E-waste API deployment handoff

Status: deployment configuration prepared. A hosted service has not yet been verified.
Inspected source: kunalarya05/API master, commit
40b2edce0a7f2bfa3a729f819e1bc6d99713ba9a.

The deployment uses the existing classifier, class_names.json, price_integration.py
and bundled pricing engine. It does not retrain either model.

## Proposed Render service

- Repository: https://github.com/kunalarya05/API
- Branch: master
- Root directory: classification
- Runtime: Python
- Build: pip install -r requirements-deploy.txt && python build_deploy.py
- Start: gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 180 app:app
- Health check: /api/health
- Environment: PYTHON_VERSION=3.11.11, TF_NUM_INTRAOP_THREADS=1,
  TF_NUM_INTEROP_THREADS=1, OMP_NUM_THREADS=1
- CORS_ORIGINS: comma-separated website origins; defaults to * without credentials.
- Start with the free plan only. Memory suitability is unverified; do not enable
  a paid plan without the owner's approval. Measure memory after model startup.

The build downloads the v5 LFS model with a pinned SHA256, restores the six
pricing models using the existing checksum-verified restore script, and imports
the application to check both engines can load. No models are retrained.

## API contract

Set BASE_URL to the actual HTTPS service URL after deployment.

GET /api/health: returns model/pricing readiness and the eight class names.

POST /api/predict: multipart/form-data with an `image` file (maximum request 10 MiB).
Response: success, classification.predicted_category, classification.confidence
(0–1), classification.confidence_percent, classification.accepted.
If accepted is false, ask the user to retake or confirm the classification
before requesting a quote. Confidence is a model score, not guaranteed accuracy.

POST /api/recommend-price: JSON body:

```json
{"category":"PCB","confidence":0.93,"state":"Uttar Pradesh",
 "city":"Ghaziabad","quantity":1,"total_weight_kg":2,
 "channel":"authorized","unit":"auto"}
```

Optional fields: subcategory, as_of_date, channel, unit.
Use the predicted category and score from /api/predict, then collect location,
quantity and weight. The response contains result.classification and
result.pricing, including recommended_rate_inr, estimated_value_inr and the
engine's provenance, range and warning fields. Preserve those warnings in the UI:
these are estimates from the existing hybrid pricing data, not live market bids.

```bash
curl "$BASE_URL/api/health"
curl -X POST "$BASE_URL/api/predict" -F 'image=@sample.jpg'
curl -X POST "$BASE_URL/api/recommend-price" \
  -H 'Content-Type: application/json' \
  -d '{"category":"PCB","confidence":0.93,"state":"Uttar Pradesh","city":"Ghaziabad","quantity":1,"total_weight_kg":2}'
```

## Camera integration

The user's website/app opens the camera with permission and uploads a captured
frame. camera_scan.py opens the computer's local camera and is not the cloud
server entry point. For a simple mobile website upload control:

```html
<input id="photo" type="file" accept="image/*" capture="environment">
```

```javascript
async function classify(file, baseUrl) {
  const data = new FormData();
  data.append("image", file);
  const response = await fetch(`${baseUrl}/api/predict`, {
    method: "POST", body: data
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Classification failed");
  return result.classification;
}
```

For a live preview, use navigator.mediaDevices.getUserMedia on HTTPS, draw the
video frame to a canvas, and upload its JPEG blob through the same endpoint.
Native apps use their platform's camera API and multipart upload.
The backend developer can also proxy both endpoints through their backend.
No API authentication or rate limiting is included; apply those in the app's
backend/gateway before broad public use. CORS is not authentication.

## Verification and remaining gates

Python syntax and source changes have been checked locally. End-to-end model
loading, dependency installation, camera accuracy, and hosted requests remain
unverified in this environment. The revised preprocessing follows camera_scan.py
(RGB float32 0–255); validate it against known images and the training pipeline.
The v5 model's output count is checked against class_names.json at startup.

Before publishing, confirm the Render workspace. After deployment: require a live deploy, health HTTP 200, test a known
image, test the resulting quote, and inspect logs/memory. Do not hand a guessed
service URL to the backend developer.
