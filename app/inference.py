# Web app and deployment by Arshi Nisar
"""Model loading and prediction.

Privacy: this module never prints, logs or stores the text a visitor types.
"""
import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# Path relative to this file, so it works locally and on Streamlit Cloud
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REQUIRED_FILES = ["vectorizer.joblib", "encoder.joblib", "model.joblib", "model_both.joblib"]

DRUG_COLUMN = "drugName"   # must match the column name used in training
MIN_WORDS = 5
MAX_CHARS = 5000


class AppError(Exception):
    """An error whose message is safe and clear enough to show to the user."""


def load_artifacts():
    """Load the 4 saved objects once. Raises AppError with a clear message."""
    missing = [f for f in REQUIRED_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        raise AppError(
            "Model files are missing from the 'models/' folder: " + ", ".join(missing)
        )

    try:
        vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
        encoder = joblib.load(MODELS_DIR / "encoder.joblib")
        model_text = joblib.load(MODELS_DIR / "model.joblib")        # text only
        model_both = joblib.load(MODELS_DIR / "model_both.joblib")   # text + drug
    except Exception as exc:
        raise AppError(
            "The model files could not be opened (" + type(exc).__name__ + "). "
            "Most likely the scikit-learn version does not match the one used for "
            "training (scikit-learn==1.7.2), or a file is damaged."
        ) from exc

    if not np.array_equal(model_text.classes_, model_both.classes_):
        raise AppError("The two models do not have the same list of conditions.")

    drugs = sorted(str(d) for d in encoder.categories_[0])
    return {
        "vectorizer": vectorizer,
        "encoder": encoder,
        "model": model_text,
        "model_both": model_both,
        "drugs": drugs,
        "drug_set": set(drugs),
    }


def clean(text):
    """Same cleaning as the notebooks: unescape HTML, collapse spaces, strip."""
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def predict(review, drug, art, top_k=3):
    """Return {'top': [(condition, probability), ...], 'model_used': str, 'notes': [str]}.

    drug = None (or an unknown drug) -> text-only model.
    drug known                       -> text + drug model.
    """
    if not isinstance(review, str) or not review.strip():
        raise AppError("Please type or paste a review first.")

    review = clean(review[:MAX_CHARS])
    if len(review.split()) < MIN_WORDS:
        raise AppError(f"Please enter a review of at least {MIN_WORDS} words.")

    notes = []
    X_text = art["vectorizer"].transform([review])
    has_known_words = X_text.nnz > 0

    use_drug = bool(drug) and drug in art["drug_set"]
    if drug and not use_drug:
        notes.append("This drug is not in the training data, so only the text was used.")

    if use_drug:
        X_drug = art["encoder"].transform(pd.DataFrame({DRUG_COLUMN: [drug]}))
        X = hstack([X_text, X_drug]).tocsr()
        model, used = art["model_both"], "text + drug"
        if not has_known_words:
            notes.append("None of the words were recognised, so the result relies on the drug only.")
    else:
        if not has_known_words:
            raise AppError(
                "None of the words in this review were recognised by the model. "
                "It was trained on English reviews. Try more words or choose a drug."
            )
        X, model, used = X_text, art["model"], "text only"

    proba = model.predict_proba(X)[0]
    order = np.argsort(-proba)[:top_k]
    top = [(str(model.classes_[i]), float(proba[i])) for i in order]
    return {"top": top, "model_used": used, "notes": notes}