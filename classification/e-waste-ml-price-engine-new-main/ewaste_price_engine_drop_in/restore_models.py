"""Restore the exact model artifacts from the versioned bundle parts."""

import hashlib
import io
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "model_bundle"
DESTINATION = ROOT / "price_engine" / "models"
EXPECTED_SHA256 = "2e60e3ad9576b257e59850bc3ede470f625ebc189d50d4b17c7b4aafddb0133b"
EXPECTED_FILES = {
    "per_kg__authorized__best.joblib",
    "per_kg__informal__best.joblib",
    "per_kg__mandi__best.joblib",
    "per_piece__authorized__best.joblib",
    "per_piece__informal__best.joblib",
    "per_piece__mandi__best.joblib",
}


def restore():
    parts = sorted(BUNDLE.glob("part-???"))
    if len(parts) != 28 or [p.name for p in parts] != [f"part-{i:03d}" for i in range(28)]:
        raise ValueError("Incomplete model bundle: expected parts 000 through 027")
    archive = b"".join(part.read_bytes() for part in parts)
    if hashlib.sha256(archive).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Model bundle checksum mismatch")

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        names = {Path(m.name).name for m in members}
        if names != EXPECTED_FILES or len(members) != len(EXPECTED_FILES):
            raise ValueError("Unexpected model files in bundle")
        if any(m.name != f"models/{Path(m.name).name}" for m in members):
            raise ValueError("Unexpected model path in bundle")

        DESTINATION.mkdir(parents=True, exist_ok=True)
        for member in members:
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read {member.name}")
            (DESTINATION / Path(member.name).name).write_bytes(source.read())
    print(f"Restored {len(members)} verified model files to {DESTINATION}")


if __name__ == "__main__":
    restore()
