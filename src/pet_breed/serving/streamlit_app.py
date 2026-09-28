"""
Interactive Streamlit Application for Pet Breed Classification & MLOps Exploration.
Can communicate with the live FastAPI endpoint or execute local PyTorch inference.
"""

import io
from pathlib import Path
from PIL import Image
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Pet Breed AI Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🐾 Pet Breed Classification & Calibrated Serving")
st.markdown(
    """
    **Production Computer Vision MLOps System** classifying 37 cat and dog breeds with **calibrated selective abstention**.
    """
)

# Sidebar: Service & Config
with st.sidebar:
    st.header("⚙️ Service Configuration")
    api_endpoint = st.text_input("FastAPI Endpoint", value=API_URL)
    
    # Check Backend Health
    health_status = "🔴 Disconnected"
    try:
        r = requests.get(f"{api_endpoint}/health", timeout=2)
        if r.status_code == 200:
            health_data = r.json()
            health_status = f"🟢 Healthy ({health_data.get('model_version', 'v1.0.0')})"
    except Exception:
        health_status = "🔴 API Offline (Start FastAPI server)"

    st.markdown(f"**Backend Status:** {health_status}")

    st.markdown("---")
    st.markdown("### 📚 Fast Links")
    st.markdown(f"- [Interactive API Docs (Swagger)]({api_endpoint}/docs)")
    st.markdown(f"- [Prometheus Metrics]({api_endpoint}/metrics)")

# Main Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload / Select Pet Photo")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png", "webp"]
    )

    # Preset Sample Selector
    sample_options = {
        "Select a sample...": None,
        "🐱 Bengal Cat": "Bengal_2.jpg",
        "🐶 Beagle": "beagle_2.jpg",
        "🐱 Persian Cat": "Persian_2.jpg",
        "🐶 Boxer": "boxer_2.jpg",
        "🐱 Siamese Cat": "Siamese_2.jpg",
    }
    selected_sample = st.selectbox("Or choose a pre-loaded test sample:", list(sample_options.keys()))

    image_to_predict = None

    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Uploaded Image", use_container_width=True)
    elif selected_sample and sample_options[selected_sample]:
        sample_filename = sample_options[selected_sample]
        sample_path = Path("data/corrupted/brightness_shift_s1") / sample_filename
        if sample_path.exists():
            image_to_predict = Image.open(sample_path)
            st.image(image_to_predict, caption=f"Sample: {selected_sample}", use_container_width=True)
        else:
            st.warning("Sample image not found on local path.")

    predict_button = st.button("🚀 Classify Breed", type="primary", use_container_width=True, disabled=image_to_predict is None)

with col2:
    st.subheader("2. Prediction & Calibration Output")
    if predict_button and image_to_predict is not None:
        with st.spinner("Running calibrated model inference..."):
            buf = io.BytesIO()
            # Convert to RGB to ensure clean transport
            if image_to_predict.mode != "RGB":
                rgb_img = image_to_predict.convert("RGB")
            else:
                rgb_img = image_to_predict
            rgb_img.save(buf, format="JPEG")
            buf.seek(0)

            try:
                response = requests.post(
                    f"{api_endpoint}/predict",
                    files={"file": ("input.jpg", buf, "image/jpeg")},
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    breed = data["breed"].replace("_", " ").title()
                    species = "🐱 Cat" if data["species"] == "cat" else "🐶 Dog"
                    conf = data["confidence"]
                    decision = data["decision"]

                    # Decision Alert
                    if decision == "confident":
                        st.success(f"### Result: **{breed}** ({species})")
                        st.info(f"**Decision:** `confident` — Calibrated Confidence: **{conf * 100:.1f}%**")
                    else:
                        st.warning(f"### Result: **{breed}** ({species})")
                        st.error(f"**Decision:** `uncertain` (Abstained) — Calibrated Confidence: **{conf * 100:.1f}%** (Below Threshold)")

                    # Top 3 Candidates
                    st.markdown("#### 📊 Top-3 Ranked Candidates")
                    for cand in data.get("top_3", []):
                        cand_breed = cand["breed"].replace("_", " ").title()
                        cand_species = "🐱" if cand["species"] == "cat" else "🐶"
                        cand_conf = cand["confidence"]
                        st.write(f"**{cand_species} {cand_breed}** ({cand_conf * 100:.1f}%)")
                        st.progress(float(cand_conf))

                    st.caption(f"Model: {data.get('model_version')} | Threshold: 0.65")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Run `uvicorn pet_breed.serving.app:app --port 8000` first.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.info("👈 Upload a pet image or select a quick test sample, then click **Classify Breed**.")
