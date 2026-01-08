import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Load trained model
# -------------------------
# For demo, we train model inside app (small dataset)
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
def label_over_reliance(row):
    if row["ai_queries_per_day"] > 12 and row["manual_research_time"] < 20 and row["copy_paste_ratio"] > 0.7:
        return 2
    elif row["ai_queries_per_day"] > 6:
        return 1
    else:
        return 0
df["over_reliance_level"] = df.apply(label_over_reliance, axis=1)

# Features
df['trust_score'] = df['copy_paste_ratio'] / (df['follow_up_questions'] + 1)
df['verification_ratio'] = df['manual_research_time'] / (df['ai_queries_per_day'] + 1)
df['dependency_index'] = df['trust_score'] * df['confidence_in_ai'] / (df['disagreement_rate'] + 0.1)

X = df[['ai_queries_per_day','follow_up_questions','copy_paste_ratio','manual_research_time',
        'confidence_in_ai','disagreement_rate','trust_score','verification_ratio','dependency_index']]
y = df['over_reliance_level']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# -------------------------
# Streamlit UI
# -------------------------
st.title("AI Over-Reliance Risk Detector 🔥")

st.markdown("""
Enter your AI usage behaviour below and get a **risk score**.
""")

# User inputs
ai_queries_per_day = st.number_input("AI Queries per Day", min_value=0, max_value=50, value=5)
follow_up_questions = st.number_input("Follow-Up Questions", min_value=0, max_value=20, value=3)
copy_paste_ratio = st.slider("Copy-Paste Ratio (0-1)", min_value=0.0, max_value=1.0, value=0.5)
manual_research_time = st.number_input("Manual Research Time (minutes)", min_value=0, max_value=500, value=30)
confidence_in_ai = st.slider("Confidence in AI (1-5)", min_value=1, max_value=5, value=3)
disagreement_rate = st.slider("Disagreement Rate (0-1)", min_value=0.0, max_value=1.0, value=0.3)

# Derived features
trust_score = copy_paste_ratio / (follow_up_questions + 1)
verification_ratio = manual_research_time / (ai_queries_per_day + 1)
dependency_index = trust_score * confidence_in_ai / (disagreement_rate + 0.1)

input_df = pd.DataFrame({
    'ai_queries_per_day':[ai_queries_per_day],
    'follow_up_questions':[follow_up_questions],
    'copy_paste_ratio':[copy_paste_ratio],
    'manual_research_time':[manual_research_time],
    'confidence_in_ai':[confidence_in_ai],
    'disagreement_rate':[disagreement_rate],
    'trust_score':[trust_score],
    'verification_ratio':[verification_ratio],
    'dependency_index':[dependency_index]
})

# Predict
prediction = rf.predict(input_df)[0]
risk_dict = {0:"Low Risk 🟢", 1:"Medium Risk 🟡", 2:"High Risk 🔴"}

st.subheader("Over-Reliance Risk Level:")
st.write(risk_dict[prediction])

# Optional explanation
st.subheader("Recommendation:")
if prediction == 2:
    st.write("⚠️ You are heavily relying on AI. Try verifying answers and reducing blind trust.")
elif prediction == 1:
    st.write("⚠️ Moderate reliance. Keep checking AI outputs and use your own judgment.")
else:
    st.write("✅ Low reliance. Good balance between AI and your own thinking.")
