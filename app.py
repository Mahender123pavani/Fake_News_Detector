import streamlit as st
import pickle
import re

# Page config
st.set_page_config(
    page_title="News_ID Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

# Load YOUR exact model files
@st.cache_resource
def load_models():
    """Load News_ID trained models"""
    model = pickle.load(open('News_ID_model.pkl', 'rb'))
    vectorizer = pickle.load(open('News_ID_vectorizer.pkl', 'rb'))
    return model, vectorizer

# Initialize models
try:
    model, vectorizer = load_models()
    st.success("✅ Models loaded successfully!")
except:
    st.error("❌ Model files not found. Check News_ID_model.pkl and News_ID_vectorizer.pkl")
    st.stop()

# Header
st.title("🛡️ News_ID Fake News Detector")
col1, col2 = st.columns(2)
with col1:
    st.metric("📊 Total Articles", "300,000")
with col2:
    st.metric("🔥 Fake News", "116,541 (38%)")

st.markdown("---")

# Input form
st.subheader("📝 Analyze News Article")
col1, col2 = st.columns([3, 1])

with col1:
    news_title = st.text_input(
        "News Title", 
        placeholder="Enter news headline..."
    )
with col2:
    news_source = st.text_input(
        "Source", 
        placeholder="e.g., CNN, BBC"
    )

news_text = st.text_area(
    "Full News Text", 
    height=200,
    placeholder="Paste complete article here..."
)

# Predict button
col1, col2 = st.columns([4, 1])
if col1.button("🔍 DETECT FAKE NEWS", type="primary", use_container_width=True):
    if news_title or news_text:
        # Combine all text (matches training)
        full_text = f"{news_title} {news_source} {news_text}".strip()
        
        # Clean text (EXACT same as training)
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', full_text.lower())
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if len(cleaned_text) < 10:
            st.warning("⚠️ Please enter more text for accurate prediction")
        else:
            # Predict using YOUR News_ID models
            vectorized = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]
            
            # Display results
            col_result, col_conf, col_prob = st.columns([2, 1, 1])
            
            with col_result:
                if prediction == 1:
                    st.error("❌ **FAKE NEWS**")
                else:
                    st.success("✅ **REAL NEWS**")
            
            with col_conf:
                confidence = max(probabilities) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col_prob:
                st.info(f"Fake Prob: {probabilities[1]*100:.1f}%")
                
    else:
        st.warning("⚠️ Please enter news title or text!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    **Built by Laxmi Prasanna** | B.Tech CSE | Data Science Portfolio<br>
    Trained on 300K News_ID dataset | Python + Scikit-learn + Streamlit
</div>
""")
