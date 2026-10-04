import streamlit as st
import pandas as pd
import pickle

# ============================================================
# RANDOM FOREST CLASSIFICATION - STREAMLIT APP
# Model: random(1).pkl
# ============================================================

st.set_page_config(
    page_title="Random Forest Prediction",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- CSS -------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero {
        padding: 28px 32px;
        border-radius: 20px;
        background: linear-gradient(135deg, #172554, #2563eb);
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.20);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin: 0;
        font-size: 38px;
        font-weight: 800;
    }

    .hero p {
        margin: 8px 0 0 0;
        font-size: 16px;
        opacity: 0.92;
    }

    .card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #172554;
        margin-bottom: 12px;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 12px;
        height: 48px;
        font-size: 16px;
        font-weight: 700;
        border: none;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #4338ca);
        color: white;
    }

    .result-yes {
        padding: 20px;
        border-radius: 15px;
        background: #ecfdf5;
        border: 1px solid #86efac;
        color: #166534;
        text-align: center;
        font-size: 25px;
        font-weight: 800;
    }

    .result-no {
        padding: 20px;
        border-radius: 15px;
        background: #fff7ed;
        border: 1px solid #fdba74;
        color: #9a3412;
        text-align: center;
        font-size: 25px;
        font-weight: 800;
    }

    .info-box {
        padding: 15px;
        border-radius: 12px;
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        color: #1e3a8a;
        margin-top: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- LOAD MODEL ---------------------------
@st.cache_resource
def load_model():
    with open("random(1).pkl", "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except Exception as e:
    st.error("Model file could not be loaded.")
    st.code(str(e))
    st.stop()


# ------------------ LABEL ENCODING MAPS ----------------------
# These mappings match the alphabetical LabelEncoder-style
# encoding used by the supplied Random Forest model.

gender_map = {
    "Female": 0,
    "Male": 1
}

marital_map = {
    "Married": 0,
    "Prefer not to say": 1,
    "Single": 2
}

occupation_map = {
    "Employee": 0,
    "House wife": 1,
    "Self Employeed": 2,
    "Student": 3
}

income_map = {
    "10001 to 25000": 0,
    "25001 to 50000": 1,
    "Below Rs.10000": 2,
    "More than 50000": 3,
    "No Income": 4
}

education_map = {
    "Graduate": 0,
    "Ph.D": 1,
    "Post Graduate": 2,
    "School": 3,
    "Uneducated": 4
}

customer_type_map = {
    "Frequent": 0,
    "New": 1,
    "Regular": 2
}


# ------------------------- HEADER ----------------------------
st.markdown("""
<div class="hero">
    <h1>🌲 Random Forest Prediction</h1>
    <p>Enter customer details below to generate a Yes / No prediction.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown("## 🌲 Model Information")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Prediction:** Yes / No")
    st.write("**Input Features:** 8")
    st.markdown("---")
    st.info(
        "Categorical fields are displayed as dropdown menus and "
        "automatically converted to the numeric values expected by the model."
    )


# --------------------- INPUT SECTION -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Customer Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        list(gender_map.keys())
    )

    marital_status = st.selectbox(
        "Marital Status",
        list(marital_map.keys())
    )

    occupation = st.selectbox(
        "Occupation",
        list(occupation_map.keys())
    )

with col2:
    monthly_income = st.selectbox(
        "Monthly Income",
        list(income_map.keys())
    )

    education = st.selectbox(
        "Educational Qualifications",
        list(education_map.keys())
    )

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    customer_type = st.selectbox(
        "Customer Type",
        list(customer_type_map.keys())
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- PREDICTION ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔮 Generate Prediction</div>', unsafe_allow_html=True)

predict_clicked = st.button("🚀 Predict")

if predict_clicked:
    try:
        # Create input in EXACTLY the same feature order as the model
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender_map[gender],
            "Marital Status": marital_map[marital_status],
            "Occupation": occupation_map[occupation],
            "Monthly Income": income_map[monthly_income],
            "Educational Qualifications": education_map[education],
            "Family size": family_size,
            "Customer Type": customer_type_map[customer_type]
        }])

        prediction = model.predict(input_data)[0]

        # Handle both string and numeric prediction formats
        if str(prediction).lower() in ["1", "yes"]:
            result = "Yes"
        else:
            result = "No"

        st.markdown("### Prediction Result")

        if result == "Yes":
            st.markdown(
                '<div class="result-yes">✅ Prediction: YES</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-no">❌ Prediction: NO</div>',
                unsafe_allow_html=True
            )

        # Probability, when available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            class_labels = [str(c) for c in model.classes_]
            probability_df = pd.DataFrame({
                "Class": class_labels,
                "Probability": probabilities
            })

            st.markdown("### 📊 Prediction Probability")
            st.dataframe(
                probability_df.style.format(
                    {"Probability": "{:.2%}"}
                ),
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            '<div class="info-box">Prediction generated successfully using the supplied Random Forest model.</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("Prediction failed. Please check the model and input configuration.")
        st.exception(e)

st.markdown("</div>", unsafe_allow_html=True)


# ------------------------ FOOTER -----------------------------
st.markdown(
    "<p style='text-align:center;color:#64748b;margin-top:25px;'>"
    "Random Forest ML Deployment • Streamlit"
    "</p>",
    unsafe_allow_html=True
)
