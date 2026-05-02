import streamlit as st
import requests
from PIL import Image
import io
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080/classify")

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️", layout="wide")

st.title("♻️ Smart Waste Classifier")
st.markdown("""
Welcome to the Smart Waste Classifier! Upload an image of a waste item, and our AI will classify it (Organic vs. Recyclable) 
and provide you with specific disposal instructions and an upcycling idea based on SDG 12 guidelines.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.markdown("---")
    
    if st.button("Classify Waste"):
        with st.spinner("Analyzing image and generating sustainability advice..."):
            try:
                # Prepare the file for sending
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send request to FastAPI backend
                response = requests.post(BACKEND_URL, files=files)
                response.raise_for_status()
                
                data = response.json()
                
                # Display Results in a Clean Dashboard
                st.subheader("Classification Result")
                
                # Determine color based on classification
                class_color = "green" if data["classification"] == "Organic" else "blue"
                st.markdown(f"**<h2 style='color: {class_color};'>{data['classification']} Waste</h2>**", unsafe_allow_html=True)
                
                st.subheader("Sustainability Advice (GenAI)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info("**Decomposition Timeline**")
                    st.write(data["decomposition_timeline"])
                    
                with col2:
                    st.success("**Recycling Instructions**")
                    st.write(data["recycling_instructions"])
                    
                with col3:
                    st.warning("**Upcycling Idea**")
                    st.write(data["upcycling_idea"])
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with the backend: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
