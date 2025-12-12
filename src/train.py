import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datasets import load_dataset

# --------------------------
# Load + preprocess
# --------------------------

ds = load_dataset("ag_news")

X_train = [x.lower() for x in ds["train"]["text"]]
y_train = ds["train"]["label"]

X_test = [x.lower() for x in ds["test"]["text"]]
y_test = ds["test"]["label"]


# --------------------------
# Vectorizer
# --------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# --------------------------
# Model
# --------------------------
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)

# --------------------------
# Evaluation
# --------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
