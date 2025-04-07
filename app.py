import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import hog, local_binary_pattern

# Set page config FIRST
st.set_page_config(page_title="Tomato Disease Detector", page_icon="🍅")

@st.cache_resource
def load_models():
    try:
        return (
            joblib.load("scaler.pkl"),
            joblib.load("lda.pkl"),
            joblib.load("ensemble_model.pkl"),
            joblib.load("label_encoder.pkl")
        )
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # HOG features
    hog_feat = hog(gray, pixels_per_cell=(8,8), 
                  cells_per_block=(3,3), 
                  block_norm='L2-Hys')
    
    # Color histogram features
    hist_feat = cv2.calcHist([image], [0,1,2], None, 
                           [16,16,16], [0,256]*3).flatten()
    
    # LBP features
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), 
                              bins=np.arange(0, 24 + 3), 
                              range=(0, 24 + 2))
    lbp_feat = lbp_hist.astype("float")
    lbp_feat /= (lbp_feat.sum() + 1e-6)
    
    # Ensure all features are 1D arrays
    return np.concatenate([
        hog_feat.ravel(),
        hist_feat.ravel(),
        lbp_feat.ravel()
    ])

def main():
    st.title("🍅 Tomato Leaf Disease Classifier")
    
    scaler, lda, model, le = load_models()
    
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"])

    if uploaded_file:
        try:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid image file")
                
            img = cv2.resize(img, (128, 128))
            features = extract_features(img)
            
            # Reshape and transform features
            features = scaler.transform(features.reshape(1, -1))
            features = lda.transform(features)
            
            # Make prediction
            pred = model.predict(features)[0]
            disease = le.inverse_transform([pred])[0]
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, channels="BGR", use_container_width=True)
            with col2:
                st.subheader("Diagnosis")
                if "healthy" in disease.lower():
                    st.success(f"✅ {disease}")
                else:
                    st.error(f"⚠️ {disease}")
                    st.info("""
                    **Recommended actions:**
                    - Isolate affected plants
                    - Consult agricultural expert
                    - Use appropriate treatments
                    """)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    with st.sidebar:
        st.header("Supported Diseases")
        st.markdown("""
        - Bacterial Spot
        - Early Blight
        - Late Blight
        - Leaf Mold
        - Septoria Leaf Spot
        - Spider Mites
        - Target Spot
        - Mosaic Virus
        - Yellow Leaf Curl Virus
        - Healthy leaves
        """)

    st.markdown("---")
    st.caption("Note: This is an AI tool. Always verify with a plant expert.")

if __name__ == "__main__":
    main()