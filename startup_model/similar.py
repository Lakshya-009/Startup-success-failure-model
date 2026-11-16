"""Similar failed / successful startups for a description (TF-IDF cosine similarity; swap for embeddings later)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

_FIELDS = ["name", "industry_raw", "country", "founded", "funding_m", "description", "failure_reason",
           "lesson", "failed_year", "exit_reason", "valuation_b"]

def _py(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v

class SimilarIndex:
    def __init__(self, df: pd.DataFrame):
        self.meta = df.reset_index(drop=True)
        docs = [self._doc(r) for r in self.meta.itertuples()]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform(docs)

    @staticmethod
    def _doc(r):
        parts = [r.industry_raw, r.description, r.failure_reason, r.lesson, r.name]
        tok = f"bucket_{r.industry}" if isinstance(r.industry, str) else ""      # shared vocabulary across both CSVs
        return " ".join(str(x) for x in parts + [tok] if isinstance(x, str) and x.strip())

    def query(self, text: str, industry: str | None = None, k_each: int = 4, min_sim: float = 0.02) -> dict:
        q = " ".join(x for x in [text or "", f"bucket_{industry}" if industry else ""] if x)
        out = {"failed": [], "succeeded": []}
        if not q.strip():
            return out
        sims = linear_kernel(self.vec.transform([q]), self.mat).ravel()
        labels = self.meta.label.values
        for label, key in [(0, "failed"), (1, "succeeded")]:
            for i in [i for i in np.argsort(-sims) if labels[i] == label and sims[i] >= min_sim][:k_each]:
                r = self.meta.iloc[i]
                out[key].append({f: _py(r[f]) for f in _FIELDS} | {"similarity": round(float(sims[i]), 3)})
        return out
