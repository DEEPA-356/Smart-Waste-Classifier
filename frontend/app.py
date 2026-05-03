import streamlit as st
import requests
from PIL import Image
import io
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080/classify")

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️", layout="wide")

# Custom CSS for Glassmorphism & Modern Aesthetic
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.4);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #004d40;
        font-weight: 700;
    }
    
    /* Metrics / Progress labels */
    .metric-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00695c;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #00695c, #00897b);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 105, 92, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 105, 92, 0.6);
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
    st.markdown("## Project Info")
    st.info("The Smart Waste Classifier uses MobileNetV2 and NVIDIA Triton to accurately classify waste, coupled with Google's Gemini to provide actionable sustainability advice.")
    
    st.markdown("## How it Works")
    st.markdown("""
    1. **Upload** an image of a waste item.
    2. **AI Analysis** classifies it as Organic or Recyclable.
    3. **GenAI** fetches decomposition timelines and upcycling tips.
    """)
    
    st.markdown("---")
    st.markdown("### Developer Credentials")
    st.markdown("**SRM Institute of Science and Technology**")
    st.markdown("*AI & ML Department*")

# --- Main Content ---
st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
st.title("♻️ Smart Waste Classifier")
st.markdown("""
Welcome to the next-generation AI waste management system! Upload an image below, and our model will instantly classify it and generate personalized sustainability advice in line with SDG 12.
""")
st.markdown("</div>", unsafe_allow_html=True)

# --- Upload Section ---
st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Waste Image Here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col_img, col_btn = st.columns([1, 1])
    
    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True, clamp=True)
        
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🚀 Analyze Waste"):
            with st.spinner("Processing through NVIDIA Triton and GenAI..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(BACKEND_URL, files=files)
                    response.raise_for_status()
                    data = response.json()
                    
                    st.session_state['classification_data'] = data
                except requests.exceptions.RequestException as e:
                    st.error(f"Error communicating with the backend: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# --- Results Section ---
if 'classification_data' in st.session_state:
    data = st.session_state['classification_data']
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown("## 📊 Classification Results")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        class_color = "#388E3C" if data["classification"] == "Organic" else "#1976D2"
        st.markdown(f"<h3 style='color: {class_color}; font-size: 2.5rem;'>{data['classification']} Waste</h3>", unsafe_allow_html=True)
        
    with res_col2:
        confidence = data.get("confidence", 0.0)
        st.markdown(f"<span class='metric-label'>AI Confidence Score: {confidence*100:.1f}%</span>", unsafe_allow_html=True)
        st.progress(float(confidence))
        
    st.markdown("---")
    st.markdown("### 🌍 Sustainability Advice")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.info("⏱️ **Decomposition Timeline**\n\n" + data.get("decomposition_timeline", "N/A"))
        
    with info_col2:
        st.success("♻️ **Recycling Instructions**\n\n" + data.get("recycling_instructions", "N/A"))
        
    with info_col3:
        st.warning("💡 **Upcycling Idea**\n\n" + data.get("upcycling_idea", "N/A"))
        
    st.markdown("</div>", unsafe_allow_html=True)
