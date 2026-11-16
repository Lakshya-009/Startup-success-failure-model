"""StartupRiskModel: the object the backend uses.

    from startup_model import StartupRiskModel
    model = StartupRiskModel.load()
    model.predict({"funding_usd_millions": 0.2, "country": "India", "industry": "edu"})
"""
import json
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from . import config
from .cleaning import bucket, load_clean, norm_country, training_frame, _isnull

BAND_LABELS = {"failure_like": "closer to failed startups", "mixed": "mixed signals", "success_like": "closer to successful startups"}


def build_pipeline(num=None, cat=None, C=config.LOGREG_C) -> Pipeline:
    """Logistic regression on log-funding (spline) + country + industry; class-balanced weights."""
    num = config.NUM_FEATURES if num is None else num
    cat = config.CAT_FEATURES if cat is None else cat
    parts = []
    if num:
        parts.append(("n", Pipeline([("imp", SimpleImputer(strategy="median")),
                                     ("spl", SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="constant")),
                                     ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("c", OneHotEncoder(handle_unknown="ignore", min_frequency=8), cat))
    return Pipeline([("pre", ColumnTransformer(parts)),
                     ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])


class StartupRiskModel:
    def __init__(self, pipeline, funding_ref, card: dict):
        self.pipeline = pipeline
        self.funding_ref = np.sort(np.asarray(funding_ref, dtype=float))
        self.card = card
        self._index = None

    # ------------------------------------------------------------------ loading / training
    @classmethod
    def load(cls, auto_retrain: bool = True) -> "StartupRiskModel":
        """Load the shipped model. If it was saved with a different scikit-learn version (or fails to load) it is rebuilt
        in memory from the bundled data (about a second), so a version mismatch never breaks the backend."""
        try:
            bundle = joblib.load(config.MODEL_PATH)
            card = json.loads(config.CARD_PATH.read_text(encoding="utf-8"))
            if card.get("sklearn_version") != sklearn.__version__:
                raise RuntimeError(f"model saved with scikit-learn {card.get('sklearn_version')}, running {sklearn.__version__}")
            return cls(bundle["pipeline"], bundle["funding_ref"], card)
        except Exception as e:
            if not auto_retrain:
                raise
            warnings.warn(f"startup_model: rebuilding model in memory ({e})")
            return cls.retrain(save=False)

    @classmethod
    def retrain(cls, save: bool = False, card: dict | None = None) -> "StartupRiskModel":
        df = training_frame(load_clean())
        X, y = df[config.NUM_FEATURES + config.CAT_FEATURES], df.label.values
        pipe = build_pipeline().fit(X, y)
        if card is None:
            try:
                card = json.loads(config.CARD_PATH.read_text(encoding="utf-8"))
            except Exception:
                card = {"model_version": config.MODEL_VERSION}
            card = {**card, "sklearn_version": sklearn.__version__, "rebuilt_locally": True}
        model = cls(pipe, df.funding_m.values, card)
        if save:
            config.ARTIFACTS.mkdir(exist_ok=True)
            joblib.dump({"pipeline": pipe, "funding_ref": model.funding_ref}, config.MODEL_PATH)
            config.CARD_PATH.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
        return model

    # ------------------------------------------------------------------ input handling
    def _prepare(self, features: dict):
        feats = features if isinstance(features, dict) else {}
        warns = []
        fund = feats.get("funding_usd_millions")
        if _isnull(fund) or isinstance(fund, bool):
            fund = None
        else:
            try:
                fund = float(fund)
            except (TypeError, ValueError):
                warns.append("funding_usd_millions was not a number and was ignored")
                fund = None
        if fund is not None and (not np.isfinite(fund) or fund < 0):
            warns.append("funding_usd_millions must be a non-negative number and was ignored")
            fund = None
        if fund is not None and fund > 500_000:
            warns.append("funding_usd_millions is unusually large; the value must be in millions of US dollars")
        country = norm_country(feats.get("country"))
        industry = bucket(feats.get("industry") if not _isnull(feats.get("industry")) else feats.get("industry_bucket"))
        country = country if isinstance(country, str) else None
        industry = industry if isinstance(industry, str) else None
        return {"funding_usd_millions": fund, "country": country, "industry": industry}, warns

    def _lookup_rate(self, score: float):
        for b in self.card.get("calibration_bins", []):
            if b["low"] <= score <= b["high"]:
                return b["success_rate"]
        return None

    # ------------------------------------------------------------------ prediction
    def predict_many(self, rows: list) -> list:
        prepared = [self._prepare(r) for r in rows]
        frame = pd.DataFrame([{
            "log_funding": np.log1p(p["funding_usd_millions"]) if p["funding_usd_millions"] is not None else np.nan,
            "country": p["country"], "industry": p["industry"]} for p, _ in prepared],
            columns=config.NUM_FEATURES + config.CAT_FEATURES)
        scores = self.pipeline.predict_proba(frame)[:, 1] if len(frame) else []
        out = []
        for (p, warns), score in zip(prepared, scores):
            score = float(score)
            missing = [k for k, v in p.items() if v is None]
            pct = None if p["funding_usd_millions"] is None else \
                float(np.searchsorted(self.funding_ref, p["funding_usd_millions"], side="right") / len(self.funding_ref))
            out_of_range = pct is not None and pct < config.OUT_OF_RANGE_PERCENTILE
            confidence = "low" if (len(missing) >= 2 or out_of_range) else "medium" if missing else "moderate"   # never 'high'
            band = "failure_like" if score < config.BAND_LOW else "success_like" if score > config.BAND_HIGH else "mixed"
            notes = list(warns)
            if out_of_range:
                notes.append("Funding is below ~90% of the startups the model learned from, so the comparison is weak; read the band loosely.")
            if missing:
                notes.append("Missing inputs: " + ", ".join(missing) + ". Providing them makes the score more reliable.")
            if len(missing) == len(p):                      # nothing usable was provided -> don't fabricate a score
                out.append({"success_score": None, "band": "insufficient_data", "band_label": "not enough information",
                            "confidence": "none", "dataset_success_rate": None, "missing_fields": missing, "out_of_range": False,
                            "funding_percentile": None, "normalized_input": p,
                            "notes": notes, "model_version": self.card.get("model_version", config.MODEL_VERSION)})
                continue
            out.append({
                "success_score": round(score, 3),
                "band": band,
                "band_label": BAND_LABELS[band],
                "confidence": confidence,
                "dataset_success_rate": self._lookup_rate(score),
                "missing_fields": missing,
                "out_of_range": bool(out_of_range),
                "funding_percentile": None if pct is None else round(pct, 3),
                "normalized_input": p,
                "notes": notes,
                "model_version": self.card.get("model_version", config.MODEL_VERSION),
            })
        return out

    def predict(self, features: dict) -> dict:
        """features: {'funding_usd_millions': float|None, 'country': str|None, 'industry': str|None}. Extra keys are ignored."""
        return self.predict_many([features])[0]

    # ------------------------------------------------------------------ similar startups (for the consultant tier)
    def similar_startups(self, text: str, industry: str | None = None, k_each: int = 4) -> dict:
        """Most similar failed / succeeded startups for a free-text description. Built lazily on first call."""
        if self._index is None:
            from .similar import SimilarIndex
            self._index = SimilarIndex(load_clean())
        ind = bucket(industry) if industry else None
        return self._index.query(text, ind if isinstance(ind, str) else None, k_each)
