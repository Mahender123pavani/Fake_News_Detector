import streamlit as st
import pickle
import re
from datetime import datetime
import pandas as pd

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="News_ID Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    model = pickle.load(open("News_ID_model.pkl", "rb"))
    vectorizer = pickle.load(open("News_ID_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()

# ===============================
# Session State (History)
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# Header
# ===============================
st.markdown("## 🛡️ News_ID Fake News Detector")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Total Articles", "300,000")
with col2:
    st.metric("🔥 Fake News", "116,541", "38%")
with col3:
    st.metric("🧠 Model Accuracy", "92%")

st.markdown("---")

# ===============================
# Input Section
# ===============================
st.markdown("### 📝 Analyze News Article")

col1, col2 = st.columns([3, 1])

with col1:
    news_title = st.text_input(
        "News Title",
        value="Government to Shut Down Internet for 7 Days Starting Monday"
    )

with col2:
    news_source = st.text_input(
        "Source",
        value="DailyNationNow.com"
    )

news_text = st.text_area(
    "Full News Text",
    height=260,
    value=(
        "The central government has reportedly decided to shut down internet "
        "services across the country for seven days starting this Monday. "
        "According to anonymous sources, the decision was taken to control "
        "misinformation and maintain national security."
    )
)

# ===============================
# Prediction
# ===============================
if st.button("🔍 DETECT FAKE NEWS", type="primary", use_container_width=True):

    if news_title.strip() or news_text.strip():

        full_text = f"{news_title} {news_source} {news_text}"

        # Same preprocessing as training
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", full_text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        fake_prob = prob[1] * 100
        real_prob = prob[0] * 100
        confidence = max(fake_prob, real_prob)

        label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"

        st.markdown("---")
        col_result, col_conf, col_prob = st.columns([2, 1, 1])

        # ===============================
        # Result
        # ===============================
        with col_result:
            if prediction == 1:
                st.error("❌ **FAKE NEWS**")
            else:
                st.success("✅ **REAL NEWS**")

        # ===============================
        # Auto-Colored Prediction Confidence
        # ===============================
        with col_conf:
            if confidence >= 80:
                st.success(f"🟢 Prediction Confidence: {confidence:.1f}%")
            elif confidence >= 60:
                st.warning(f"🟡 Prediction Confidence: {confidence:.1f}%")
            else:
                st.error(f"🔴 Prediction Confidence: {confidence:.1f}%")

        # ===============================
        # Probabilities
        # ===============================
        with col_prob:
            st.write(f"**Likelihood of Fake News:** {fake_prob:.1f}%")
            st.write(f"**Likelihood of Real News:** {real_prob:.1f}%")

        # ===============================
        # Borderline Warning
        # ===============================
        if confidence < 60:
            st.warning(
                "⚠️ Borderline prediction — the article shows a mix of "
                "factual language and sensational cues."
            )

        st.caption(
            "🔍 This prediction is probabilistic and should not be treated as definitive proof."
        )

        # ===============================
        # Save to History
        # ===============================
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Title": news_title,
            "Source": news_source,
            "Prediction": label,
            "Prediction Confidence (%)": round(confidence, 1),
            "Fake Probability (%)": round(fake_prob, 1),
            "Real Probability (%)": round(real_prob, 1)
        })

    else:
        st.warning("⚠️ Please enter a news title or article text!")

# ===============================
# History Section
# ===============================
st.markdown("---")
st.markdown("### 📈 History of Analyzed Articles")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df_history.to_csv(index=False)
        st.download_button(
            label="📤 Export History to CSV",
            data=csv_data,
            file_name=f"fake_news_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.success("History cleared!")

else:
    st.info("No articles analyzed yet.")

# ===============================
# Footer
# ===============================
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: #666; font-size: 14px;'>
        <b>Built by Laxmi Prasanna</b> | B.Tech CSE | Data Science Portfolio<br>
        Trained on 300K+ News Articles | Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
