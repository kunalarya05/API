"""Run from classification after installing requirements-deploy.txt."""
import hashlib
from pathlib import Path
import runpy
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "best_e_waste_model_v5.keras"
EXPECTED = "196accc02888d1fb62229939d70a89945b124bf7579cd9b69f7edd1d1ddbdc1d"
SIZE = 24679057
# Pin to the inspected commit, so a future model cannot silently change.
URL = ("https://media.githubusercontent.com/media/kunalarya05/API/"
       "40b2edce0a7f2bfa3a729f819e1bc6d99713ba9a/classification/"
       "best_e_waste_model_v5.keras")

def valid(path):
    return (path.exists() and path.stat().st_size == SIZE
            and hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED)

if not valid(MODEL):
    temporary = MODEL.with_suffix(".download")
    try:
        with urllib.request.urlopen(URL, timeout=120) as response:
            temporary.write_bytes(response.read(SIZE + 1))
        if not valid(temporary):
            raise RuntimeError("Classifier download size or checksum mismatch")
        temporary.replace(MODEL)
    finally:
        temporary.unlink(missing_ok=True)

runpy.run_path(str(ROOT / "e-waste-ml-price-engine-new-main" /
                   "ewaste_price_engine_drop_in" / "restore_models.py"),
               run_name="__main__")

# Fail the build if either model cannot load with these dependencies.
import app  # noqa: E402
assert app.model is not None and app.engine is not None
print("Classifier and pricing engine loaded successfully")
