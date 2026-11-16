"""Cleaning + harmonising the two raw CSVs into one labelled table; also the runtime normalisers for incoming values."""
import re
import numpy as np
import pandas as pd
from .config import DATA_DIR, INDUSTRIES

COUNTRY = {"usa": "US", "united states": "US", "u.s.": "US", "us": "US", "uk": "UK", "united kingdom": "UK", "england": "UK",
           "china": "China", "india": "India", "germany": "Germany", "france": "France", "canada": "Canada", "israel": "Israel",
           "singapore": "Singapore", "south korea": "South Korea", "brazil": "Brazil", "australia": "Australia",
           "sweden": "Sweden", "indonesia": "Indonesia", "japan": "Japan", "netherlands": "Netherlands"}

BUCKETS = [
    ("fintech", r"fintech|financ|payment|bank|lending|insur|crypto|blockchain|bnpl|neobank|wealth|trading|bitcoin|defi|web3"),
    ("health", r"health|bio|pharma|medic|care|drug|genom|diagnos|fitness|wellness"),
    ("edu", r"edtech|educat|learning|tutor|school"),
    ("mobility_hw", r"mobility|transport|auto|vehicle|ev\b|electric|scooter|ride|aviation|drone|robot|hardware|energy|solar|battery|space|aerospace|cleantech|semiconductor|manufactur|3d|device|electronics"),
    ("commerce_logistics", r"commerce|marketplace|retail|grocery|delivery|logistic|supply|fashion|consumer|food|travel|hospitality|real estate|proptech|furniture|beauty|fmcg|shipping|freight"),
    ("software_ai", r"software|saas|ai\b|artificial|data|cloud|cyber|security|analytic|tech|internet|platform|enterprise|developer|it\b|information|network|telecom|communication"),
    ("media_social", r"media|social|game|gaming|entertainment|music|video|content|messaging|streaming|adtech|advertis|market"),
]

def _isnull(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())

def norm_country(c):
    """Free text -> one of the training vocabulary (e.g. 'USA'/'United States' -> 'US'); unlisted countries -> 'Other'; empty -> NaN."""
    if _isnull(c):
        return np.nan
    return COUNTRY.get(str(c).split("/")[0].split(",")[0].strip().lower(), "Other")

def bucket(s):
    """Free-text industry -> bucket by keyword; empty -> NaN. A value already in INDUSTRIES is returned as is."""
    if _isnull(s):
        return np.nan
    s = str(s).strip().lower()
    if s in INDUSTRIES:
        return s
    return next((b for b, p in BUCKETS if re.search(p, s)), "other")

def money_m(label):
    """'$1.1B+' / '€16M' / '$808.5M' -> USD millions. Unparseable ('Undisclosed', '$209B in assets') -> NaN."""
    if _isnull(label):
        return np.nan
    s = str(label).replace(",", "")
    if re.search(r"assets|undisclosed|unknown|no data|internal", s, re.I):
        return np.nan
    m = re.search(r"([\d.]+)\s*([KMB])\b", s)
    return float(m.group(1)) * {"K": 1e-3, "M": 1, "B": 1e3}[m.group(2)] if m else np.nan

def load_clean() -> pd.DataFrame:
    """Label: 1 = became a unicorn / exited, 0 = failed. Resolves duplicates, overlaps and unit errors (see README)."""
    u = pd.read_csv(DATA_DIR / "unicorn_companies.csv")
    f = pd.read_csv(DATA_DIR / "startup-failures.csv")
    f["key"] = f.name.str.lower().str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
    u["key"] = u.Company.str.lower().str.strip()
    f = f.drop_duplicates("key")                                   # duplicate names inside failures file
    overlap = set(u.key) & set(f.key)                              # listed in both files -> failures file wins
    reason = u.Exit_Reason.fillna("").str.lower()
    u = u[~u.key.isin(overlap) & ~reason.isin(["devalued", "devaluation"])].copy()   # drop contradictory / ambiguous
    u["label"] = np.where(u.Exit_Reason.fillna("").str.lower().isin(["defunct", "bankruptcy"]), 0, 1)
    f["label"] = 0
    f["funding_m"] = f.funding_raised_label.map(money_m).fillna(
        f.funding_raised_usd_millions.where(f.funding_raised_usd_millions < 1e5))   # re-parse: raw column has unit errors

    fa = pd.DataFrame({
        "name": f.name, "label": 0, "founded": f.founded, "country": f.country.map(norm_country),
        "industry": f.industry.map(bucket), "industry_raw": f.industry, "funding_m": f.funding_m,
        "description": f.lesson, "failure_reason": f.failure_reason, "lesson": f.lesson,
        "failed_year": f.failed, "exit_reason": np.nan, "valuation_b": np.nan, "src": "failures"})
    ua = pd.DataFrame({
        "name": u.Company, "label": u.label, "founded": u.Founded_Year, "country": u.Country.map(norm_country),
        "industry": u.Industry.map(bucket), "industry_raw": u.Industry, "funding_m": u.Total_Funding_USD_Millions,
        "description": u.Products, "failure_reason": np.nan, "lesson": np.nan,
        "failed_year": np.nan, "exit_reason": u.Exit_Reason, "valuation_b": u.Valuation_USD_Billions, "src": "unicorns"})
    df = pd.concat([fa, ua], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)   # shuffled
    df["log_funding"] = np.log1p(df.funding_m)
    return df

def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Only rows where every model input is really observed, so that 'which cells are empty' can't leak the data source."""
    return df.dropna(subset=["funding_m", "country", "industry"]).reset_index(drop=True)
