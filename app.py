import streamlit as st
import pandas as pd
import pickle

# ==============================
# LOAD MODEL
# ==============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="AI Over-Reliance Detector", layout="centered")
st.title("🤖 AI Over-Reliance Risk Detector")

st.write("Analyze your AI usage behaviour and assess dependency risk.")

st.markdown("---")

# ==============================
# USER INPUTS
# ==============================
ai_queries_per_day = st.number_input("AI Queries per Day", 0, 50, 5)
follow_up_questions = st.number_input("Follow-Up Questions", 0, 20, 3)
copy_paste_ratio = st.slider("Copy-Paste Ratio (0–1)", 0.0, 1.0, 0.5)
manual_research_time = st.number_input("Manual Research Time (minutes)", 0, 500, 30)
confidence_in_ai = st.slider("Confidence in AI (1–5)", 1, 5, 3)
disagreement_rate = st.slider("Disagreement Rate (0–1)", 0.0, 1.0, 0.3)

# ==============================
# FEATURE ENGINEERING
# ==============================
trust_score = copy_paste_ratio / (follow_up_questions + 1)
verification_ratio = manual_research_time / (ai_queries_per_day + 1)
dependency_index = trust_score * confidence_in_ai / (disagreement_rate + 0.2)

input_df = pd.DataFrame({
    "ai_queries_per_day": [ai_queries_per_day],
    "follow_up_questions": [follow_up_questions],
    "copy_paste_ratio": [copy_paste_ratio],
    "manual_research_time": [manual_research_time],
    "confidence_in_ai": [confidence_in_ai],
    "disagreement_rate": [disagreement_rate],
    "trust_score": [trust_score],
    "verification_ratio": [verification_ratio],
    "dependency_index": [dependency_index]
})

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Check Risk Level"):
    prediction = model.predict(input_df)[0]

    result_map = {
        0: "🟢 Low Risk",
        1: "🟡 Medium Risk",
        2: "🔴 High Risk"
    }

    st.subheader("Result")
    st.success(result_map[prediction])

    st.subheader("Recommendation")
    if prediction == 2:
        st.warning("Reduce blind trust in AI. Verify outputs independently.")
    elif prediction == 1:
        st.info("Moderate reliance. Maintain a healthy balance.")
    else:
        st.success("Healthy AI usage pattern.")

st.markdown("⚠️ Educational use only")
