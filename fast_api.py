"""
ML Project FastAPI Backend
Serves both Classification (Credit Risk) and Regression (House Prices) models.
Models are loaded from pre-trained .pkl files in the models/ folder.

Run with:
    uvicorn fast_api:app --reload --port 8000
"""

import io
import os
import pickle
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="ML Project API",
    description="Credit Risk Classification + House Prices Regression",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

# Explicit OPTIONS handler - must be registered BEFORE any other routes
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Access-Control-Max-Age": "86400",
        }
    )

# ═══════════════════════════════════════════════════════════════
# GLOBAL STATE — loaded from pkl files
# ═══════════════════════════════════════════════════════════════
clf_models   = {}
clf_scaler   = None
clf_features = []
clf_results  = {}

reg_pipelines        = {}
reg_results          = {}
reg_numeric_cols     = []
reg_categorical_cols = []

# Plot style
BG = "#0f1117"; TEXT = "#ecf0f1"; GRID = "#2c3e50"
DEFAULT_CLR = "#e74c3c"; SAFE = "#2ecc71"; ACCENT = "#f39c12"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "grid.linewidth": 0.5,
})


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def load_models():
    """Load all pre-trained models from the models/ folder."""
    global clf_models, clf_scaler, clf_features, clf_results
    global reg_pipelines, reg_results, reg_numeric_cols, reg_categorical_cols

    try:
        with open(os.path.join(MODELS_DIR, "clf_models.pkl"), "rb") as f:
            clf_models = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "clf_scaler.pkl"), "rb") as f:
            clf_scaler = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "clf_features.pkl"), "rb") as f:
            clf_features = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "clf_results.pkl"), "rb") as f:
            clf_results = pickle.load(f)
        print("[OK] Classification models loaded from pkl.")
    except FileNotFoundError as e:
        print(f"[ERROR] Classification pkl not found: {e}")
        print("  → Run save_models.py locally first, then upload the models/ folder.")

    try:
        with open(os.path.join(MODELS_DIR, "reg_pipelines.pkl"), "rb") as f:
            reg_pipelines = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "reg_results.pkl"), "rb") as f:
            reg_results = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "reg_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
            reg_numeric_cols     = meta["numeric_cols"]
            reg_categorical_cols = meta["categorical_cols"]
        print("[OK] Regression models loaded from pkl.")
    except FileNotFoundError as e:
        print(f"[ERROR] Regression pkl not found: {e}")
        print("  → Run save_models.py locally first, then upload the models/ folder.")


# ═══════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup_event():
    load_models()
    print("Startup complete.")


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════
class ClassificationInput(BaseModel):
    person_age: float = 30
    person_income: float = 60000
    person_home_ownership: str = "RENT"
    person_emp_length: float = 5
    loan_intent: str = "PERSONAL"
    loan_grade: str = "B"
    loan_amnt: float = 10000
    loan_int_rate: float = 12.0
    loan_percent_income: float = 0.15
    cb_person_default_on_file: str = "N"
    cb_person_cred_hist_length: float = 4


class RegressionInput(BaseModel):
    OverallQual: int = 7
    GrLivArea: float = 1500
    GarageCars: int = 2
    TotalBsmtSF: float = 800
    FullBath: int = 2
    YearBuilt: int = 2000
    YearRemodAdd: int = 2005
    LotArea: float = 8500
    MSSubClass: int = 20
    Neighborhood: str = "NAmes"


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════
@app.get("/api/classification/results")
def get_classification_results():
    return {
        "models": [
            {"name": k, "accuracy": v["accuracy"]}
            for k, v in sorted(clf_results.items(), key=lambda x: -x[1]["accuracy"])
        ],
        "best_model": max(clf_results, key=lambda k: clf_results[k]["accuracy"]) if clf_results else None,
    }


@app.post("/api/classification/predict")
def predict_classification(data: ClassificationInput):
    if not clf_models:
        raise HTTPException(status_code=503, detail="Classification models not loaded. Check models/ folder.")

    raw = {
        "person_age":                 data.person_age,
        "person_income":              data.person_income,
        "person_home_ownership":      data.person_home_ownership,
        "person_emp_length":          data.person_emp_length,
        "loan_intent":                data.loan_intent,
        "loan_grade":                 data.loan_grade,
        "loan_amnt":                  data.loan_amnt,
        "loan_int_rate":              data.loan_int_rate,
        "loan_percent_income":        data.loan_percent_income,
        "cb_person_default_on_file":  data.cb_person_default_on_file,
        "cb_person_cred_hist_length": data.cb_person_cred_hist_length,
    }

    cat_map = {
        "person_home_ownership":     {"RENT": 3, "OWN": 2, "MORTGAGE": 0, "OTHER": 1},
        "loan_intent":               {"PERSONAL": 4, "EDUCATION": 1, "MEDICAL": 3, "VENTURE": 5,
                                      "HOMEIMPROVEMENT": 2, "DEBTCONSOLIDATION": 0},
        "loan_grade":                {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
        "cb_person_default_on_file": {"N": 0, "Y": 1},
    }

    row = {}
    base_features = [
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
        "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
    ]
    for f in base_features:
        val = raw[f]
        if f in cat_map:
            val = cat_map[f].get(str(val).upper(), 0)
        row[f] = float(val)

    row["debt_to_income"]      = data.loan_amnt / (data.person_income + 1)
    row["high_interest"]       = float(data.loan_int_rate > 15)
    row["income_per_emp_year"] = data.person_income / (data.person_emp_length + 1)

    ordered = {k: row.get(k, 0.0) for k in clf_features} if clf_features else row
    X_in = np.array([list(ordered.values())])
    if clf_scaler:
        X_in = clf_scaler.transform(X_in)

    predictions = {}
    for name, mdl in clf_models.items():
        pred = int(mdl.predict(X_in)[0])
        prob = None
        if hasattr(mdl, "predict_proba"):
            prob = round(float(mdl.predict_proba(X_in)[0][1]) * 100, 2)
        predictions[name] = {"prediction": pred, "default_probability_pct": prob}

    names_list = list(predictions.keys())
    probs_list = [v["default_probability_pct"] or (100 if v["prediction"] else 0)
                  for v in predictions.values()]
    colors = [DEFAULT_CLR if p > 50 else SAFE for p in probs_list]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names_list, probs_list, color=colors, edgecolor="none", height=0.5)
    ax.axvline(50, color=ACCENT, linestyle="--", linewidth=1.5, alpha=0.6, label="50% threshold")
    for bar, val in zip(bars, probs_list):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color=TEXT, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Default Probability (%)")
    ax.set_title("Default Risk Prediction — All Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    chart_b64 = fig_to_b64(fig)

    best = max(clf_models, key=lambda k: clf_results.get(k, {}).get("accuracy", 0))
    return {
        "input_summary": raw,
        "predictions":   predictions,
        "best_model":    best,
        "verdict":       "DEFAULT RISK" if predictions[best]["prediction"] == 1 else "SAFE",
        "chart_base64":  chart_b64,
    }


@app.get("/api/classification/charts")
def classification_charts():
    if not clf_results:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    names = list(clf_results.keys())
    accs  = [clf_results[n]["accuracy"] for n in names]
    best  = max(clf_results, key=lambda k: clf_results[k]["accuracy"])
    colors = [ACCENT if n == best else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor="none", height=0.5)
    for bar, val in zip(bars, accs):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", color=TEXT, fontweight="bold", fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Classification Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.axvline(accs[names.index(best)], color=ACCENT, linestyle="--", alpha=0.4)
    ax.grid(axis="x", alpha=0.2)
    chart_b64 = fig_to_b64(fig)
    return {"chart_base64": chart_b64, "models": clf_results, "best_model": best}


# ═══════════════════════════════════════════════════════════════
# REGRESSION ENDPOINTS
# ═══════════════════════════════════════════════════════════════
@app.get("/api/regression/results")
def get_regression_results():
    return {
        "models": [
            {"name": k, **v}
            for k, v in sorted(reg_results.items(), key=lambda x: -x[1]["r2"])
        ],
        "best_model": max(reg_results, key=lambda k: reg_results[k]["r2"]) if reg_results else None,
    }


@app.post("/api/regression/predict")
def predict_regression(data: RegressionInput):
    if not reg_pipelines:
        raise HTTPException(status_code=503, detail="Regression models not loaded. Check models/ folder.")

    input_dict = data.dict()
    predictions = {}
    for name, pipe in reg_pipelines.items():
        try:
            X_in = pd.DataFrame([input_dict])
            pred = float(pipe.predict(X_in)[0])
        except Exception:
            nums = [input_dict.get(c, 0) for c in reg_numeric_cols]
            X_in = np.array([nums])
            pred = float(pipe.predict(X_in)[0])
        predictions[name] = round(pred, 2)

    names_list = list(predictions.keys())
    preds_list  = list(predictions.values())
    best = max(reg_results, key=lambda k: reg_results[k]["r2"]) if reg_results else names_list[0]
    bar_colors = [ACCENT if n == best else "#3498db" for n in names_list]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    bars = axes[0].bar(names_list, preds_list, color=bar_colors, edgecolor="none", width=0.5)
    for bar, val in zip(bars, preds_list):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
                     f"${val:,.0f}", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Predicted Sale Price ($)")
    axes[0].set_title("Predicted House Price — All Models", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.2)

    if reg_results:
        r2_names = list(reg_results.keys())
        r2_vals  = [reg_results[n]["r2"] for n in r2_names]
        r2_colors = [ACCENT if n == best else "#9b59b6" for n in r2_names]
        axes[1].bar(r2_names, r2_vals, color=r2_colors, edgecolor="none", width=0.5)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_ylabel("R² Score")
        axes[1].set_title("Model R² Comparison", fontsize=11, fontweight="bold")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].grid(axis="y", alpha=0.2)

    plt.tight_layout()
    chart_b64 = fig_to_b64(fig)

    best_pred = predictions.get(best, list(predictions.values())[0])
    return {
        "input_summary": input_dict,
        "predictions":   predictions,
        "best_model":    best,
        "best_prediction_usd": best_pred,
        "chart_base64":  chart_b64,
    }


@app.get("/api/regression/charts")
def regression_charts():
    if not reg_results:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    names = list(reg_results.keys())
    best  = max(reg_results, key=lambda k: reg_results[k]["r2"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("r2", "R² Score", True), ("rmse", "RMSE ($)", False), ("mae", "MAE ($)", False)]

    for ax, (metric, label, higher_better) in zip(axes, metrics):
        vals   = [reg_results[n][metric] for n in names]
        colors = [ACCENT if n == best else "#3498db" for n in names]
        bars   = ax.barh(names, vals, color=colors, edgecolor="none", height=0.5)
        for bar, val in zip(bars, vals):
            ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}" if metric == "r2" else f"${val:,.0f}",
                    va="center", color=TEXT, fontsize=9, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_title(f"{label}\n({'Higher' if higher_better else 'Lower'} is better)",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)

    plt.suptitle("Regression Model Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart_b64 = fig_to_b64(fig)
    return {"chart_base64": chart_b64, "models": reg_results, "best_model": best}


# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "classification_models": list(clf_models.keys()),
        "regression_models":     list(reg_pipelines.keys()),
    }


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True)
