import streamlit as st
import pickle
import re

# Load YOUR trained News_ID models
@st.cache_resource
def load_models():
    model = pickle.load(open('News_ID_model.pkl', 'rb'))
    vectorizer = pickle.load(open('News_ID_vectorizer.pkl', 'rb'))
    return model, vectorizer

# Load models
model, vectorizer = load_models()

# Page config
st.set_page_config(
    page_title="News_ID Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

# Header
st.title("🛡️ News_ID Fake News Detector")
st.markdown("**🚀 Trained on YOUR 300K News_ID dataset**")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Articles", "300,000")
with col2:
    st.metric("Fake Detected", "116,541 (38%)")

st.markdown("---")

# Input section
st.subheader("📝 Enter News to Analyze")
col1, col2 = st.columns([2, 1])

with col1:
    news_title = st.text_input("**News Title:**", placeholder="Enter news headline...")
with col2:
    news_source = st.text_input("**Source:**", placeholder="Optional")

news_text = st.text_area(
    "**Full News Text:**", 
    height=200,
    placeholder="Paste the complete news article here..."
)

# Prediction button
if st.button("🔍 DETECT FAKE NEWS", type="primary", use_container_width=True):
    if news_title or news_text:
        # Combine inputs (same as training)
        full_news = (news_title + " " + news_source + " " + news_text).strip()
        
        # Clean text (EXACTLY like training)
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', full_news.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Predict using YOUR News_ID model
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        
        # Results
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if prediction == 1:
                st.error("❌ **FAKE NEWS DETECTED**")
            else:
                st.success("✅ **REAL NEWS**")
        
        with col2:
            confidence = max(probability) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col3:
            st.info(f"**Fake Prob:** {probability[1]*100:.1f}%")
        
        st.markdown("---")
        
    else:
        st.warning("⚠️ Please enter news title OR text!")

# Footer
st.markdown("---")
st.markdown("""
**Built by Laxmi Prasanna** | **B.Tech CSE** | **Data Science Portfolio**

**Tech Stack:**
- Python + Scikit-learn (Random Forest)
- NLP (TF-IDF Vectorization) 
- Streamlit Web App
- 300K News_ID Dataset
""")
