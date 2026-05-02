# ============================================================
# HEART DISEASE PREDICTOR — Streamlit Web App
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# ─── Page Configuration ────────────────────────────────────
st.set_page_config(
    page_title="🫀 Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #e74c3c;
        font-size: 2.5rem;
        font-weight: 700;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box-positive {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-box-negative {
        background: linear-gradient(135deg, #27ae60, #1e8449);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("breast_cancer_detector.pickle", "rb"))
        return model
    except FileNotFoundError:
        st.error("❌ Modèle non trouvé. Assurez-vous que 'breast_cancer_detector.pickle' est dans le même dossier.")
        return None

model = load_model()

# ─── Header ────────────────────────────────────────────────
st.markdown('''<div class="main-title">🫀 Heart Disease Predictor</div>''', unsafe_allow_html=True)
st.markdown('''<div class="subtitle">Outil d\'aide au diagnostic cardiaque basé sur le Machine Learning</div>''', unsafe_allow_html=True)

st.info("⚠️ **Avertissement médical :** Cet outil est à titre informatif uniquement et ne remplace pas un avis médical professionnel.")

st.divider()

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
    st.title("📋 À propos")
    st.markdown("""
    **Heart Disease Predictor** utilise des algorithmes de Machine Learning 
    entraînés sur le *Heart Failure Prediction Dataset* (918 patients).
    
    **Features utilisées :**
    - Données démographiques (âge, sexe)
    - Paramètres cliniques (cholestérol, ECG, etc.)
    - Résultats d'effort (MaxHR, Oldpeak, etc.)
    
    **Performance du modèle :**
    - Accuracy : > 85%
    - Optimisé pour minimiser les faux négatifs (Recall)
    """)
    
    st.divider()
    st.markdown("**Développé avec :** Python · Scikit-learn · Streamlit")

# ─── Input Form ────────────────────────────────────────────
st.subheader("📝 Entrez les données du patient")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Données Démographiques**")
    age = st.slider("Âge (années)", min_value=20, max_value=90, value=50, step=1)
    sex = st.selectbox("Sexe", options=["Masculin (M)", "Féminin (F)"])
    sex_encoded = 1 if "Masculin" in sex else 0
    
    fasting_bs = st.selectbox("Glycémie à jeun > 120 mg/dl ?", 
                               options=["Non (0)", "Oui (1)"])
    fasting_bs_val = 0 if "Non" in fasting_bs else 1

with col2:
    st.markdown("**🩺 Paramètres Cliniques**")
    resting_bp = st.number_input("Pression artérielle au repos (mm Hg)", 
                                  min_value=60, max_value=220, value=120)
    cholesterol = st.number_input("Cholestérol (mg/dl)", 
                                   min_value=100, max_value=600, value=200)
    resting_ecg = st.selectbox("ECG au repos", 
                                options=["Normal", "ST", "LVH"])
    
with col3:
    st.markdown("**🏃 Paramètres d\'Effort**")
    max_hr = st.slider("Fréquence cardiaque max (bpm)", 
                        min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Angine à l\'effort ?", 
                                    options=["Non (N)", "Oui (Y)"])
    exercise_angina_val = 0 if "Non" in exercise_angina else 1
    oldpeak = st.slider("Oldpeak (dépression ST)", 
                         min_value=-3.0, max_value=7.0, value=0.0, step=0.1)

st.markdown("**💓 Type de Douleur et Pente ST**")
col4, col5 = st.columns(2)
with col4:
    chest_pain = st.selectbox("Type de douleur thoracique", 
                               options=["ATA (Angine Atypique)", "NAP (Non Anginale)", 
                                        "ASY (Asymptomatique)", "TA (Angine Typique)"])
with col5:
    st_slope = st.selectbox("Pente du segment ST", 
                             options=["Up (Montante)", "Flat (Plate)", "Down (Descendante)"])

st.divider()

# ─── Preprocess & Predict ──────────────────────────────────
def preprocess_input(age, sex_encoded, resting_bp, cholesterol, fasting_bs_val,
                     resting_ecg, max_hr, exercise_angina_val, oldpeak, 
                     chest_pain, st_slope, feature_names):
    """Prétraite les inputs selon le même pipeline que l\'entraînement."""
    
    # Valeurs de base pour toutes les features
    input_dict = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs_val,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_encoded': sex_encoded,
        'ExerciseAngina_encoded': exercise_angina_val,
        # ChestPainType (drop_first=True → ATA est la référence)
        'ChestPainType_NAP': 1 if 'NAP' in chest_pain else 0,
        'ChestPainType_ASY': 1 if 'ASY' in chest_pain else 0,
        'ChestPainType_TA':  1 if 'TA (' in chest_pain else 0,
        # RestingECG (drop_first=True → LVH est la référence)
        'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
        'RestingECG_ST':     1 if resting_ecg == 'ST' else 0,
        # ST_Slope (drop_first=True → Down est la référence)
        'ST_Slope_Flat': 1 if 'Flat' in st_slope else 0,
        'ST_Slope_Up':   1 if 'Up' in st_slope else 0,
    }
    
    # Créer un DataFrame avec toutes les features dans le bon ordre
    df_input = pd.DataFrame([input_dict])
    
    # Sélectionner uniquement les features du modèle
    available_features = [f for f in feature_names if f in df_input.columns]
    df_input = df_input.reindex(columns=feature_names, fill_value=0)
    
    return df_input

# Bouton de prédiction
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_btn = st.button("🔍 Analyser le Risque Cardiaque", use_container_width=True)

if predict_btn and model is not None:
    # Récupérer les features attendues par le modèle
    try:
        feature_names = model.feature_names_in_.tolist()
    except AttributeError:
        # Si le modèle n\'a pas feature_names_in_, utiliser une liste par défaut
        feature_names = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 
                         'MaxHR', 'Oldpeak', 'Sex_encoded', 'ExerciseAngina_encoded',
                         'ChestPainType_NAP', 'ChestPainType_ASY', 'ChestPainType_TA',
                         'RestingECG_Normal', 'RestingECG_ST', 'ST_Slope_Flat', 'ST_Slope_Up']
    
    df_input = preprocess_input(age, sex_encoded, resting_bp, cholesterol, 
                                 fasting_bs_val, resting_ecg, max_hr, 
                                 exercise_angina_val, oldpeak, chest_pain, 
                                 st_slope, feature_names)
    
    prediction = model.predict(df_input)[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_input)[0]
        confidence = proba[1] if prediction == 1 else proba[0]
        risk_pct = proba[1] * 100
    else:
        confidence = None
        risk_pct = None
    
    st.divider()
    st.subheader("🔬 Résultat de l\'Analyse")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if prediction == 1:
            st.markdown('''<div class="result-box-positive">
                ❤️‍🔥 RISQUE ÉLEVÉ DE MALADIE CARDIAQUE<br>
                <small>Une consultation médicale urgente est recommandée</small>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''<div class="result-box-negative">
                💚 FAIBLE RISQUE DE MALADIE CARDIAQUE<br>
                <small>Continuez à maintenir un mode de vie sain</small>
            </div>''', unsafe_allow_html=True)
    
    with res_col2:
        if risk_pct is not None:
            st.metric("🎯 Probabilité de Maladie", f"{risk_pct:.1f}%")
            
            # Jauge de risque visuelle
            if risk_pct < 30:
                color = "green"
                label = "Faible risque"
            elif risk_pct < 60:
                color = "orange"
                label = "Risque modéré"
            else:
                color = "red"
                label = "Risque élevé"
            
            st.markdown(f"**Niveau de risque :** :{color}[{label}]")
            st.progress(int(risk_pct))
    
    # Récapitulatif des données saisies
    st.subheader("📊 Récapitulatif des Données Patient")
    recap_col1, recap_col2, recap_col3 = st.columns(3)
    
    with recap_col1:
        st.markdown(f"""
        <div class="metric-card">
        <b>👤 Démographie</b><br>
        Âge : {age} ans<br>
        Sexe : {"Masculin" if sex_encoded==1 else "Féminin"}<br>
        Glycémie à jeun : {"Élevée" if fasting_bs_val==1 else "Normale"}
        </div>
        """, unsafe_allow_html=True)
    
    with recap_col2:
        st.markdown(f"""
        <div class="metric-card">
        <b>🩺 Clinique</b><br>
        Tension : {resting_bp} mm Hg<br>
        Cholestérol : {cholesterol} mg/dl<br>
        ECG : {resting_ecg}
        </div>
        """, unsafe_allow_html=True)
    
    with recap_col3:
        st.markdown(f"""
        <div class="metric-card">
        <b>🏃 Effort</b><br>
        MaxHR : {max_hr} bpm<br>
        Angine effort : {"Oui" if exercise_angina_val==1 else "Non"}<br>
        Oldpeak : {oldpeak}
        </div>
        """, unsafe_allow_html=True)
    
    st.caption("⚠️ Ce résultat est fourni à titre indicatif uniquement. Consultez toujours un professionnel de santé pour un diagnostic médical.")

elif predict_btn and model is None:
    st.error("❌ Impossible de charger le modèle. Vérifiez que 'breast_cancer_detector.pickle' existe.")

# ─── Footer ────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style=\"text-align: center; color: #95a5a6; padding: 1rem;\">
    🫀 Heart Disease Predictor | Machine Learning — CRISP-DM | Python · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
