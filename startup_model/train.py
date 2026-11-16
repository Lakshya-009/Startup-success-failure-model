"""Train, evaluate and save the model.      python -m startup_model.train

Writes: startup_model/artifacts/model.joblib, model_card.json, reports/metrics.json, reports/report.md, reports/evaluation.png
"""
import json
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, learning_curve, train_test_split

from . import config
from .cleaning import load_clean, training_frame
from .model import build_pipeline

warnings.filterwarnings("ignore")
FEATS = config.NUM_FEATURES + config.CAT_FEATURES
BIN_EDGES = [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]


def fold_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"auc": roc_auc_score(y, p), "balanced_acc": balanced_accuracy_score(y, pred), "accuracy": accuracy_score(y, pred),
            "f1_success": f1_score(y, pred), "brier": brier_score_loss(y, p)}


def summarize(rows):
    df = pd.DataFrame(rows)
    return {c: {"mean": round(df[c].mean(), 4), "std": round(df[c].std(), 4)} for c in df.columns}


def repeated_cv(X, y, num=None, cat=None, n_repeats=3):
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=config.RANDOM_STATE)
    rows, oof, cnt = [], np.zeros(len(y)), np.zeros(len(y))
    for tr, te in rs.split(X, y):
        p = build_pipeline(num, cat).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        rows.append(fold_metrics(y[te], p)); oof[te] += p; cnt[te] += 1
    return summarize(rows), oof / cnt


def main():
    full = load_clean()
    df = training_frame(full)
    X, y = df[FEATS], df.label.values
    print(f"clean rows={len(full)}  training rows (all inputs observed)={len(df)}  successes={int(y.sum())}  failures={int((1 - y).sum())}")

    # 1) repeated 5-fold CV (primary estimate) + out-of-fold scores for calibration/band checks
    cv, oof = repeated_cv(X, y)
    print("CV (5-fold x3):", {k: v["mean"] for k, v in cv.items()})

    # 2) hold-out split (train vs test gap -> over/underfit check)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    m = build_pipeline().fit(Xtr, ytr)
    ptr, pte = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]
    holdout = {"train": {k: round(v, 4) for k, v in fold_metrics(ytr, ptr).items()},
               "test": {k: round(v, 4) for k, v in fold_metrics(yte, pte).items()},
               "test_confusion_matrix_[[TN,FP],[FN,TP]]": confusion_matrix(yte, pte >= 0.5).tolist(),
               "n_train": len(ytr), "n_test": len(yte)}

    # 3) learning curve
    sizes, trs, vals = learning_curve(build_pipeline(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE),
                                      train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], scoring="roc_auc")
    lc = [{"n_train": int(n), "train_auc": round(a, 4), "cv_auc": round(b, 4)} for n, a, b in zip(sizes, trs.mean(1), vals.mean(1))]

    # 4) what each input contributes
    ablation = {}
    for name, num, cat in [("funding only", config.NUM_FEATURES, []), ("country only", [], ["country"]), ("industry only", [], ["industry"]),
                           ("funding + country", config.NUM_FEATURES, ["country"]), ("funding + industry", config.NUM_FEATURES, ["industry"]),
                           ("all three (final)", config.NUM_FEATURES, config.CAT_FEATURES)]:
        s, _ = repeated_cv(X, y, num, cat, n_repeats=2)
        ablation[name] = {"auc": s["auc"]["mean"], "balanced_acc": s["balanced_acc"]["mean"]}

    # 5) calibration bins + band precision from out-of-fold scores
    bins = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (oof >= lo) & (oof <= hi) if hi == 1.0 else (oof >= lo) & (oof < hi)
        bins.append({"low": lo, "high": hi, "n": int(mask.sum()), "success_rate": round(float(y[mask].mean()), 3) if mask.any() else None})
    lo_m, hi_m = oof < config.BAND_LOW, oof > config.BAND_HIGH
    bands = {"failure_like": {"rule": f"score < {config.BAND_LOW}", "n": int(lo_m.sum()), "share_actually_failed": round(float(1 - y[lo_m].mean()), 3)},
             "mixed": {"rule": f"{config.BAND_LOW} <= score <= {config.BAND_HIGH}", "n": int((~lo_m & ~hi_m).sum()),
                       "share_actually_succeeded": round(float(y[~lo_m & ~hi_m].mean()), 3)},
             "success_like": {"rule": f"score > {config.BAND_HIGH}", "n": int(hi_m.sum()), "share_actually_succeeded": round(float(y[hi_m].mean()), 3)}}

    baseline = {"majority_class_accuracy": round(max(y.mean(), 1 - y.mean()), 4), "majority_class_balanced_accuracy": 0.5, "random_auc": 0.5}

    # 6) final model on all training rows -> artifacts
    final = build_pipeline().fit(X, y)
    config.ARTIFACTS.mkdir(exist_ok=True)
    funding_ref = np.sort(df.funding_m.values)
    joblib.dump({"pipeline": final, "funding_ref": funding_ref}, config.MODEL_PATH)
    card = {
        "model_version": config.MODEL_VERSION, "trained_on": str(date.today()),
        "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "algorithm": "Logistic regression (class_weight=balanced, C=%s); funding via cubic spline of log1p(USD millions); one-hot country + industry" % config.LOGREG_C,
        "label": "1 = became a unicorn / exited (IPO, acquired, still operating as unicorn); 0 = failed (shut down / bankrupt / in failures list)",
        "inputs": {"funding_usd_millions": "float >= 0, millions of USD, total raised", "country": "free text -> normalised",
                   "industry": f"one of {config.INDUSTRIES} or free text mapped by keywords"},
        "training_rows": int(len(df)), "class_counts": {"succeeded": int(y.sum()), "failed": int((1 - y).sum())},
        "cv_5fold_x3": cv, "baseline": baseline, "score_bands": bands, "calibration_bins": bins,
        "funding_percentiles_usd_m": {str(q): round(float(np.quantile(funding_ref, q)), 2) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
        "limitations": [
            "Score is a similarity-to-groups score from a class-balanced model; it is NOT the real-world probability a startup succeeds.",
            "Training data = notable/well-funded companies (unicorn list + published failures); very small startups are out of range.",
            "Country and industry signal partly reflects which kinds of companies appear in each source list.",
            "Success means unicorn/exit, not 'profitable small business'.",
            "Files are two different sources, so rows with missing fields were excluded from training to avoid learning the source.",
        ]}
    config.CARD_PATH.write_text(json.dumps(card, indent=2), encoding="utf-8")
    metrics = {"cv_5fold_x3": cv, "holdout_80_20": holdout, "learning_curve": lc, "ablation_cv_auc": ablation, "baseline": baseline,
               "bands_from_out_of_fold_scores": bands, "calibration_bins": bins}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, len(df), int(y.sum()))
    try:
        plot(y, oof, lc, bins)
    except Exception as e:
        print("plots skipped:", e)
    print("saved:", config.MODEL_PATH, "| holdout test:", holdout["test"])


def write_report(m, n, n_pos):
    cv, ho = m["cv_5fold_x3"], m["holdout_80_20"]
    L = [f"# Evaluation report (model v{config.MODEL_VERSION})", "",
         f"Training rows: {n} ({n_pos} succeeded, {n - n_pos} failed). Majority-class accuracy: {m['baseline']['majority_class_accuracy']:.1%}.", "",
         "## Primary estimate: repeated 5-fold cross-validation (3 repeats)", "", "| metric | mean | std |", "|---|---|---|"]
    L += [f"| {k} | {v['mean']:.3f} | {v['std']:.3f} |" for k, v in cv.items()]
    L += ["", "Accuracy is measured at a 0.5 threshold on a class-balanced score. Because only ~25% of rows are successes, "
          "balanced accuracy and AUC are the fairer headline numbers than raw accuracy.", "",
          "## Hold-out (80/20) and overfitting check", "", "| split | AUC | balanced acc | accuracy |", "|---|---|---|---|",
          f"| train | {ho['train']['auc']:.3f} | {ho['train']['balanced_acc']:.3f} | {ho['train']['accuracy']:.3f} |",
          f"| test | {ho['test']['auc']:.3f} | {ho['test']['balanced_acc']:.3f} | {ho['test']['accuracy']:.3f} |", "",
          f"Test confusion matrix [[TN, FP], [FN, TP]]: {ho['test_confusion_matrix_[[TN,FP],[FN,TP]]']}", "",
          "Learning curve (AUC): " + "; ".join(f"n={r['n_train']}: train {r['train_auc']:.3f} / CV {r['cv_auc']:.3f}" for r in m["learning_curve"]), "",
          "## What each input contributes (CV AUC)", "", "| inputs | AUC | balanced acc |", "|---|---|---|"]
    L += [f"| {k} | {v['auc']:.3f} | {v['balanced_acc']:.3f} |" for k, v in m["ablation_cv_auc"].items()]
    L += ["", "## Score bands (from out-of-fold scores)", ""]
    for k, v in m["bands_from_out_of_fold_scores"].items():
        L.append(f"- **{k}** ({v['rule']}): {v['n']} startups; " + ", ".join(f"{a.replace('_', ' ')} = {b:.1%}" for a, b in v.items() if a.startswith("share")))
    L += ["", "## Score vs. observed success rate in the data", "", "| score range | n | share that succeeded |", "|---|---|---|"]
    L += [f"| {b['low']:.2f}-{b['high']:.2f} | {b['n']} | {b['success_rate']:.1%} |" for b in m["calibration_bins"] if b["success_rate"] is not None]
    (config.REPORTS_DIR / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def plot(y, oof, lc, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fpr, tpr, _ = roc_curve(y, oof)
    ax[0, 0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y, oof):.3f}"); ax[0, 0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0, 0].set(title="ROC (out-of-fold)", xlabel="False positive rate", ylabel="True positive rate"); ax[0, 0].legend()
    ok = [b for b in bins if b["success_rate"] is not None]
    ax[0, 1].bar([f"{b['low']:.2f}-{b['high']:.2f}" for b in ok], [b["success_rate"] for b in ok], color="#2f5d50")
    for i, b in enumerate(ok):
        ax[0, 1].text(i, b["success_rate"] + 0.01, f"n={b['n']}", ha="center", fontsize=8)
    ax[0, 1].set(title="Observed success rate by score range", xlabel="model score", ylabel="share that succeeded"); ax[0, 1].tick_params(axis="x", labelsize=8)
    ax[1, 0].plot([r["n_train"] for r in lc], [r["train_auc"] for r in lc], "o-", label="train")
    ax[1, 0].plot([r["n_train"] for r in lc], [r["cv_auc"] for r in lc], "o-", label="cross-validation")
    ax[1, 0].set(title="Learning curve (AUC)", xlabel="training rows", ylabel="AUC"); ax[1, 0].legend()
    cm = confusion_matrix(y, oof >= 0.5)
    ax[1, 1].imshow(cm, cmap="Greens")
    for (i, j), v in np.ndenumerate(cm):
        ax[1, 1].text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax[1, 1].set(title="Confusion matrix (out-of-fold, 0.5)", xticks=[0, 1], yticks=[0, 1], xticklabels=["pred failed", "pred success"],
                 yticklabels=["actual failed", "actual success"])
    fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "evaluation.png", dpi=110)


if __name__ == "__main__":
    main()
