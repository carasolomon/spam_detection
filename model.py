import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

DATA_PATH = "data/spam.csv"
MODEL_PATH = "model_best.pkl"

def load_data():
    data = pd.read_csv(DATA_PATH, encoding="latin-1")
    if 'v1' in data.columns and 'v2' in data.columns:
        data = data[['v1', 'v2']]
        data.columns = ['label', 'message']
    data['label'] = data['label'].str.lower()
    return data

def train_and_evaluate_model(data):
    data['label_num'] = data.label.map({'ham': 0, 'spam': 1})
    X = data['message']
    y = data['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words="english")),
        ('clf', MultinomialNB())
    ])

    param_grid = {
        'tfidf__min_df': [1, 2],
        'tfidf__max_df': [0.9, 1.0],
        'clf__alpha': [0.1, 0.5, 1.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))

    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Test set performance:\n", report)
    
    return grid

def load_or_train_model():
    data = load_data()
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model = train_and_evaluate_model(data)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
    return model

if __name__ == "__main__":
    data = load_data()
    model = train_and_evaluate_model(data)