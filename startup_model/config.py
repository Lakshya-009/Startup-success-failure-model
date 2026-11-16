from pathlib import Path

PKG = Path(__file__).resolve().parent
DATA_DIR = PKG / "data"
ARTIFACTS = PKG / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
CARD_PATH = ARTIFACTS / "model_card.json"
REPORTS_DIR = PKG.parent / "reports"

MODEL_VERSION = "0.1.0"
RANDOM_STATE = 42

INDUSTRIES = ["fintech", "health", "edu", "mobility_hw", "commerce_logistics", "software_ai", "media_social", "other"]
NUM_FEATURES = ["log_funding"]
CAT_FEATURES = ["country", "industry"]
INPUT_FIELDS = ["funding_usd_millions", "country", "industry"]   # what the backend must try to supply

# Score bands (score = model output in [0,1], class-balanced, NOT a real-world probability)
BAND_LOW = 0.35
BAND_HIGH = 0.65
OUT_OF_RANGE_PERCENTILE = 0.10     # funding below the 10th percentile of the training data -> weak comparison
LOGREG_C = 0.3
