import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. DATASET CREATION
# ==============================
np.random.seed(42)
n = 300

data = {
    "ai_queries_per_day": np.random.randint(1, 20, n),
    "follow_up_questions": np.random.randint(0, 10, n),
    "copy_paste_ratio": np.round(np.random.uniform(0, 1, n), 2),
    "manual_research_time": np.random.randint(0, 120, n),
    "confidence_in_ai": np.random.randint(1, 6, n),
    "disagreement_rate": np.round(np.random.uniform(0, 1, n), 2)
}

df = pd.DataFrame(data)

# ==============================
# 2. LABEL CREATION
# ==============================
def label_over_reliance(row):
    score = (
        row["ai_queries_per_day"] * 0.3 +
        row["copy_paste_ratio"] * 10 +
        row["confidence_in_ai"] * 1.5 -
        row["manual_research_time"] * 0.05
    )
    if score > 10:
        return 2   # High
    elif score > 5:
        return 1   # Medium
    else:
        return 0   # Low

df["over_reliance_level"] = df.apply(label_over_reliance, axis=1)

# ==============================
# 3. FEATURE ENGINEERING
# ==============================
df["trust_score"] = df["copy_paste_ratio"] / (df["follow_up_questions"] + 1)
df["verification_ratio"] = df["manual_research_time"] / (df["ai_queries_per_day"] + 1)
df["dependency_index"] = (
    df["trust_score"] * df["confidence_in_ai"] /
    (df["disagreement_rate"] + 0.2)
)

X = df[
    [
        "ai_queries_per_day",
        "follow_up_questions",
        "copy_paste_ratio",
        "manual_research_time",
        "confidence_in_ai",
        "disagreement_rate",
        "trust_score",
        "verification_ratio",
        "dependency_index"
    ]
]
y = df["over_reliance_level"]

# ==============================
# 4. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==============================
# 5. MODEL TRAINING
# ==============================
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=6,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 6. EVALUATION
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 2))
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Medium", "High"]
    )
)

# ==============================
# 7. SAVE MODEL
# ==============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl saved successfully")
