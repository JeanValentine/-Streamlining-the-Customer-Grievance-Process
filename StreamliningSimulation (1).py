import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from sentence_transformers import SentenceTransformer
    SENT_TRANS_AVAILABLE = True
except Exception:
    SENT_TRANS_AVAILABLE = False

for pkg in ("stopwords", "punkt", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except Exception:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c

    low = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"\d+", " ", text)
 
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = [tok for tok in text.split() if tok not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)

def tfidf_top_n_by_class(tfidf: TfidfVectorizer, X_tfidf, labels, n=15):
    feature_names = np.array(tfidf.get_feature_names_out())
    classes = np.unique(labels)
    topn = {}
    for cls in classes:
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            topn[cls] = []
            continue
        avg = np.asarray(X_tfidf[idx].mean(axis=0)).ravel()
        top_indices = avg.argsort()[-n:][::-1]
        topn[cls] = feature_names[top_indices].tolist()
    return topn

def train_and_eval_tfidf_model(X_train_texts, X_test_texts, y_train, y_test, out_dir: Path, max_features=20000):
    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train_texts)
    X_test_tfidf = tfidf.transform(X_test_texts)
    print("TF-IDF shape (train):", X_train_tfidf.shape)

    print("Training LogisticRegression on TF-IDF features...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print("TF-IDF Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # save
    joblib.dump(tfidf, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(clf, out_dir / "tfidf_logreg.joblib")
    return tfidf, clf, X_train_tfidf, X_test_tfidf

def train_and_eval_transformer_model(texts_train, texts_test, y_train, y_test, out_dir: Path, model_name="all-MiniLM-L6-v2"):
    if not SENT_TRANS_AVAILABLE:
        print("sentence-transformers not installed; skipping transformer-embedding model.")
        return None, None
    print(f"Loading SentenceTransformer '{model_name}' (this may download a model)...")
    st = SentenceTransformer(model_name)
    X_train_emb = st.encode(texts_train, show_progress_bar=True, convert_to_numpy=True)
    X_test_emb = st.encode(texts_test, show_progress_bar=True, convert_to_numpy=True)
    print("Transformer embeddings shapes:", X_train_emb.shape, X_test_emb.shape)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_emb, y_train)
    y_pred = clf.predict(X_test_emb)
    print("Transformer-embedding Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(st, out_dir / "sentence_transformer.pkl")
    joblib.dump(clf, out_dir / "transformer_logreg.joblib")
    return st, clf

def vader_sentiment_analysis(texts: List[str]) -> pd.DataFrame:
    if SentimentIntensityAnalyzer is None:
        raise RuntimeError("vaderSentiment not installed")
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        sc = analyzer.polarity_scores("" if pd.isna(t) else str(t))
        rows.append(sc)
    return pd.DataFrame(rows)

def plot_sentiment_distribution(df_sent: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df_sent['compound'], bins=40, kde=False, ax=ax)
    ax.set_title("VADER compound score distribution")
    ax.set_xlabel("Compound score")
    fig.tight_layout()
    fig.savefig(out_dir / "vader_compound_distribution.png")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='vader_label', data=df_sent, order=['Negative','Neutral','Positive'], ax=ax)
    ax.set_title("VADER sentiment labels")
    fig.tight_layout()
    fig.savefig(out_dir / "vader_label_counts.png")
    plt.close(fig)

def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading data:", args.data)
    df = pd.read_csv(args.data, low_memory=False)
    print("Rows,Cols:", df.shape)

    text_col = args.text_col or find_column(df, ["Complaint Description", "complaint", "consumer_complaint_narrative", "narrative", "description", "Complaint"])
    if text_col is None:
        raise KeyError("Could not find complaint text column; pass --text_col <columnname>")
    print("Using text column:", text_col)

    dept_col = args.dept_col or find_column(df, ["Product", "Department", "Issue", "Sub-product", "sub_issue", "company", "Company"])
    if dept_col is None:
        raise KeyError("Could not find department/product column; pass --dept_col <columnname>")
    print("Using department column:", dept_col)


    date_col = args.date_col or find_column(df, ["Date received", "Date", "date_received", "date", "Received Date"])
    if date_col:

        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            min_d, max_d = df[date_col].min(), df[date_col].max()
            print(f"Date range in {date_col}: {min_d} to {max_d}")
            df['__year'] = df[date_col].dt.year
        except Exception as e:
            print("Could not parse date column:", e)
    else:
        print("No date column found (that's OK).")

    df[dept_col] = df[dept_col].fillna("UNKNOWN").astype(str)

    print("Preprocessing text (this may take some time)...")
    df["text_clean"] = df[text_col].fillna("").astype(str).map(preprocess_text)

    print("Sample after preprocessing:")
    print(df[[text_col, "text_clean"]].head(3).to_string())

    X = df["text_clean"].values
    y_raw = df[dept_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    print("Detected classes (departments/products):", list(class_names))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=args.test_size, stratify=y, random_state=args.random_seed
    )

    tfidf, tfidf_clf, X_train_tfidf, X_test_tfidf = train_and_eval_tfidf_model(
        X_train, X_test, y_train, y_test, out_dir, max_features=args.max_features
    )

    joblib.dump(le, out_dir / "label_encoder.joblib")

    topn = tfidf_top_n_by_class(tfidf, X_train_tfidf, y_train, n=12)
    print("Top TF-IDF words per class (sample):")
    for cls_idx, words in topn.items():
        print(f" - {le.inverse_transform([cls_idx])[0]}: {', '.join(words[:8])}")
    with open(out_dir / "top_tfidf_per_class.json", "w") as f:
        json.dump({str(le.inverse_transform([int(k)])[0]): v for k, v in topn.items()}, f, indent=2)

    if SENT_TRANS_AVAILABLE:
        st, trans_clf = train_and_eval_transformer_model(X_train.tolist(), X_test.tolist(), y_train, y_test, out_dir)
    else:
        print("sentence-transformers not available; skipping transformer-based modeling. To enable, pip install sentence-transformers")

    if SentimentIntensityAnalyzer is None:
        print("vaderSentiment not installed. Install via 'pip install vaderSentiment' to run sentiment analysis.")
    else:
        print("Running VADER sentiment analysis...")
        sent_df = vader_sentiment_analysis(df[text_col].fillna("").astype(str).tolist())
  
        df_sent = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
   
        def label_compound(c):
            if c <= -0.05:
                return "Negative"
            elif c >= 0.05:
                return "Positive"
            else:
                return "Neutral"
        df_sent["vader_label"] = df_sent["compound"].apply(label_compound)
  
        df_sent.to_csv(out_dir / "data_with_vader.csv", index=False)
    
        plot_sentiment_distribution(df_sent, out_dir)
    
        prod_sent = df_sent.groupby([dept_col, "vader_label"]).size().unstack(fill_value=0)
        print("Sentiment counts per product (sample):")
        print(prod_sent.head(10).to_string())
        prod_sent.to_csv(out_dir / "product_sentiment_counts.csv")

        neg_examples = df_sent.sort_values("compound").head(10)[[text_col, "compound", dept_col]]
        neg_examples.to_csv(out_dir / "top_negative_examples.csv", index=False)

        insights = {
            "triage_rule_examples": {
                "urgent_review": "compound <= -0.5  -> send to SLA-urgent team for manual review and remediation",
                "priority_review": "compound <= -0.25 -> escalate to product manager / complaint resolution queue",
                "monitor": "compound between -0.25 and +0.25 -> monitor aggregated changes, log for analytics",
                "positive_feedback": "compound >= 0.5 -> possible feature praise; track for PR/marketing"
            },
            "aggregations": "Aggregate average compound score by department/product and time interval (day/week/month). Departments with worsening average compound need investigation.",
            "root_cause": "Join high-negative cases with TF-IDF top words per class to identify common complaint themes (billing, delay, fraud, onboarding)."
        }
        with open(out_dir / "vader_insights.json", "w") as f:
            json.dump(insights, f, indent=2)

    print("All artifacts saved to", out_dir)
    print("Next suggestions: hyperparameter tuning, adding explainability (LIME/SHAP), and creating an automated triage rule engine using the model + VADER signals.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Grievance NLP pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to complaints CSV")
    parser.add_argument("--text_col", type=str, default=None, help="Complaint text column name (optional)")
    parser.add_argument("--dept_col", type=str, default=None, help="Department/product column name (optional)")
    parser.add_argument("--date_col", type=str, default=None, help="Date column (optional)")
    parser.add_argument("--out_dir", type=str, default="./grievance_out", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_features", type=int, default=20000, help="Max features for TF-IDF")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    main(args)