import streamlit as st
import pickle
import re

# Page configuration
st.set_page_config(
    page_title="News_ID Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

# Load YOUR News_ID models
@st.cache_resource
def load_models():
    model = pickle.load(open('News_ID_model.pkl', 'rb'))
    vectorizer = pickle.load(open('News_ID_vectorizer.pkl', 'rb'))
    return model, vectorizer

# Load models
model, vectorizer = load_models()

# Header
st.markdown("## 🛡️ News_ID Fake News Detector")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Total Articles", "300,000")
with col2:
    st.metric("🔥 Fake News", "116,541", "38%")
with col3:
    st.metric("🧠 Model Accuracy", "92%")

st.markdown("---")

# Input section
st.markdown("### 📝 Analyze News Article")
col1, col2 = st.columns([3, 1])

with col1:
    news_title = st.text_input(
        "News Title",
        placeholder="Enter news headline...",
        value="Government to Shut Down Internet for 7 Days Starting Monday"
    )
with col2:
    news_source = st.text_input(
        "Source", 
        placeholder="e.g., CNN, BBC",
        value="DailyNationNow.com"
    )

news_text = st.text_area(
    "Full News Text",
    height=250,
    placeholder="Paste complete article here...",
    value="The central government has reportedly decided to shut down internet services across the country for seven days starting this Monday. According to anonymous sources, the decision was taken to 'control misinformation and maintain national security.' Citizens have been advised to withdraw cash immediately and complete all online transactions before Sunday night."
)

# Prediction
col1, col2 = st.columns([4, 1])
if col1.button("🔍 DETECT FAKE NEWS", type="primary", use_container_width=True):
    if news_title or news_text:
        # Combine inputs
        full_text = f"{news_title} {news_source} {news_text}".strip()
        
        # Clean text (same as training)
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', full_text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Predict
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        
        st.markdown("---")
        
        col_result, col_conf, col_prob = st.columns([2, 1, 1])
        with col_result:
            if prediction == 1:
                st.error("❌ **FAKE NEWS**")
            else:
                st.success("✅ **REAL NEWS**")
        
        with col_conf:
            confidence = max(probability) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col_prob:
            st.info(f"Fake Prob: {probability[1]*100:.1f}%")
    else:
        st.warning("⚠️ Please enter news title or text!")

# Footer
st.markdown("""
<hr>
<div style='text-align: center; color: #666; font-size: 14px;'>
    <b>Built by Laxmi Prasanna</b> | B.Tech CSE | Data Science Portfolio<br>
    Trained on 300K News_ID dataset | Python • Scikit-learn • Streamlit
</div>
""", unsafe_allow_html=True)
