import streamlit as st
import numpy as np
import cv2
import joblib
import base64
from skimage.feature import graycomatrix, graycoprops

# ==============================
# 🎨 BACKGROUND + MAROON THEME
# ==============================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        /* Background */
        .stApp {{
            background: linear-gradient(rgba(81,4,0,0.75), rgba(0,0,0,0.7)),
                        url("data:image/jpg;base64,{data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Top header */
        header {{
            background-color: rgba(81, 4, 0, 0.95) !important;
        }}

        /* Upload box */
        .stFileUploader {{
            background-color: rgba(81, 4, 0, 0.6) !important;
            padding: 15px;
            border-radius: 10px;
        }}

        /* Upload label */
        .stFileUploader label {{
            color: white !important;
            font-weight: bold;
        }}

        /* Button */
        .stButton>button {{
            background-color: #510400 !important;
            color: white !important;
            border-radius: 8px;
            border: none;
        }}

        .stButton>button:hover {{
            background-color: #7a0800 !important;
            color: white;
        }}

        /* Info box */
        .stAlert {{
            background-color: rgba(81, 4, 0, 0.6) !important;
            color: white !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
set_bg("background.jpg")


# ==============================
# LOAD MODELS
# ==============================
model = joblib.load("rf_model.pkl")
selector = joblib.load("selector.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# UI HEADER
# ==============================
st.markdown(
    "<h1 style='text-align: center; color: white;'>Anemia Detection App</h1>",
    unsafe_allow_html=True
)

st.info("Upload a clear fingernail image, then click 'Detect Anemia'.")


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(img):
    img = cv2.resize(img, (224,224))

    rgb_mean = np.mean(img.reshape(-1,3), axis=0)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_mean = np.mean(hsv.reshape(-1,3), axis=0)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_mean = np.mean(lab.reshape(-1,3), axis=0)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb_mean = np.mean(ycrcb.reshape(-1,3), axis=0)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]

    features = np.concatenate([
        rgb_mean, hsv_mean, lab_mean, ycrcb_mean,
        [contrast, homogeneity, energy, correlation]
    ])

    return features


# ==============================
# UI INPUT
# ==============================
uploaded_file = st.file_uploader("Upload fingernail image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        st.error("Invalid image")
        st.stop()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Uploaded Image")

    if st.button("Detect Anemia"):
        features = extract_features(img).reshape(1,-1)

        features = selector.transform(features)
        features = scaler.transform(features)

        prob = model.predict_proba(features)[0][1]

        st.progress(int(prob * 100))
        st.write(f"Anemia Probability: {prob*100:.2f}%")

        if prob > 0.5:
            st.error("Anemic")
        else:
            st.success("Non-Anemic")