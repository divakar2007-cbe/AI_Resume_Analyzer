from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

DATA=Path(__file__).resolve().parent.parent/"data"/"resume_job_fit.csv"

MIN_ROWS_PER_CLASS = 2  # smallest count that still allows a stratified split


class MLModelError(Exception):
    """Raised when the ML pipeline cannot train/evaluate (missing, malformed,
    or insufficient dataset). Callers should catch this and show a friendly
    message instead of crashing on the underlying pandas/sklearn exception."""
    pass


def train_and_evaluate():
    if not DATA.exists():
        raise MLModelError(
            f"Dataset not found at '{DATA}'. Make sure data/resume_job_fit.csv "
            "is present before training the model."
        )

    try:
        df = pd.read_csv(DATA)
    except Exception as e:
        raise MLModelError(f"Could not read the dataset CSV ({e}).") from e

    if "text" not in df.columns or "label" not in df.columns:
        raise MLModelError(
            "Dataset is malformed: expected columns 'text' and 'label' "
            f"but found {list(df.columns)}."
        )

    df = df.dropna(subset=["text", "label"])
    if df.empty:
        raise MLModelError("Dataset has no usable rows after removing empty text/label values.")

    class_counts = df["label"].value_counts()
    if len(class_counts) < 2:
        raise MLModelError(
            "Dataset must contain at least two label classes (0 and 1) to train a classifier."
        )
    if (class_counts < MIN_ROWS_PER_CLASS).any():
        raise MLModelError(
            "Dataset is too small: each label class needs at least "
            f"{MIN_ROWS_PER_CLASS} rows to run a stratified train/test split. "
            f"Current counts: {class_counts.to_dict()}."
        )

    try:
        Xtr,Xte,ytr,yte=train_test_split(df["text"],df["label"],test_size=.30,random_state=42,stratify=df["label"])
        model=Pipeline([("tfidf",TfidfVectorizer(stop_words="english",ngram_range=(1,2))),("clf",LogisticRegression(max_iter=2000))])
        model.fit(Xtr,ytr); pred=model.predict(Xte)
    except Exception as e:
        raise MLModelError(f"Model training/evaluation failed: {e}") from e

    return {
        "accuracy":round(accuracy_score(yte,pred)*100,2),
        "precision":round(precision_score(yte,pred,zero_division=0)*100,2),
        "recall":round(recall_score(yte,pred,zero_division=0)*100,2),
        "f1":round(f1_score(yte,pred,zero_division=0)*100,2),
        "confusion_matrix":confusion_matrix(yte,pred).tolist(),
        "note":"This bundled dataset is a small demonstration dataset for verifying the ML pipeline. Do not present these metrics as production accuracy. For a research-grade result, replace it with a larger, representative labeled dataset."
    }
