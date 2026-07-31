import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------
# 1. PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Medical Summary Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------
def clean_clinical_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'([a-z])([.,!?])', r'\1 \2', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2])

def render_interactive_distribution(df):
    if 'source_condition_query' not in df.columns:
        st.error("Column 'source_condition_query' not found in dataset.")
        return

    # Aggregate counts and sort ascending for horizontal bar chart display
    counts = df['source_condition_query'].value_counts().reset_index()
    counts.columns = ['Disease Category', 'Number of Clinical Trials']
    counts = counts.sort_values(by='Number of Clinical Trials', ascending=True)

    # Build Plotly horizontal bar chart with Viridis color palette
    fig = px.bar(
        counts,
        x='Number of Clinical Trials',
        y='Disease Category',
        orientation='h',
        color='Number of Clinical Trials',
        color_continuous_scale='Viridis',
        title='Distribution of Target Disease Categories (source_condition_query)'
    )

    # Style layout to match clean aesthetic
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        xaxis_title="Number of Clinical Trials",
        yaxis_title="Disease Category"
    )

    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def load_ml_pipeline():
    try:
        return joblib.load('tfidf_vectorizer.pkl'), joblib.load('clinical_trial_classifier.pkl')
    except Exception:
        return None, None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('cleaned_clinical_data.csv')
    except Exception:
        try:
            return pd.read_csv('clinical_trials_raw_patient2trial_conditions.csv')
        except Exception:
            return pd.DataFrame({
                'source_condition_query': ['breast cancer'],
                'nct_id': ['NCT03676114'],
                'title': ['Sample Title'],
                'official_title': ['Official Title'],
                'brief_summary': ['Sample brief summary.'],
                'conditions': ['Breast Cancer'],
                'interventions': ['ketamine | Saline'],
                'overall_status': ['COMPLETED'],
                'study_type': ['INTERVENTIONAL'],
                'phase': ['PHASE4'],
                'sex': ['FEMALE'],
                'minimum_age': ['20 Years'],
                'maximum_age': ['65 Years'],
                'healthy_volunteers': [False],
                'eligibility_criteria': ['Inclusion Criteria...'],
                'clinicaltrials_url': ['https://clinicaltrials.gov/study/NCT03676114']
            })

tfidf_vec, lr_model = load_ml_pipeline()
df_data = load_dataset()
cm = joblib.load('confusion_matrix.pkl')
classes = joblib.load('model_classes.pkl')

# ----------------------------------------------------
# 1. HELPER / DUMMY PREPROCESSING (Replace with yours)
# ----------------------------------------------------
def clean_clinical_text(text):
    return text.strip()

tfidf_vec = None  # Replace with your loaded vectorizer
lr_model = None   # Replace with your loaded model

# ----------------------------------------------------
# 2. SESSION STATE
# ----------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "classifier"

def go_to_page(page_name):
    st.session_state.current_page = page_name

# ----------------------------------------------------
# 3. GLOBAL STYLING
# ----------------------------------------------------

st.markdown("""
    <style>
    /* 1. Global Page Layout & Hide Sidebar */
    [data-testid="stSidebar"] { display: none !important; }
    header[data-testid="stHeader"] { background-color: transparent !important; z-index: 100; }
    div[data-testid="stAppViewContainer"] { padding-top: 0 !important; }

    .stApp {
        background: linear-gradient(135deg, #fbcfe8 0%, #e0e7ff 50%, #bae6fd 100%) !important;
        min-height: 100vh;
    }

    .block-container {
        max-width: 950px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        margin: 0 auto !important;
    }

    /* 2. Hero Header & Badge Components */
    .project-badge-container { text-align: center; margin-bottom: 0.8rem; }
    .project-badge {
        display: inline-block; 
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.9); 
        color: #1e40af;
        font-weight: 700; 
        font-size: 0.95rem; 
        padding: 0.35rem 1.2rem;
        border-radius: 9999px; 
        letter-spacing: 0.8px;
    }
    .hero-title { font-size: 2.2rem; font-weight: 800; color: #1e293b; text-align: center; margin-bottom: 0.3rem; }
    .hero-subtitle { font-size: 0.95rem; color: #475569; text-align: center; margin-bottom: 1.5rem; }

    /* 3. Text Input Box (Rounded Pill with Natural Text Alignment) */
    .stTextInput input {
        background-color: #ffffff !important; 
        border-radius: 9999px !important;
        font-size: 1.02rem !important; 
        padding: 0.9rem 1.8rem !important; 
        text-align: left !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.06) !important;
    }

    /* 4. Container Cards (Glassmorphism Effect) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 24px !important;
        padding: 1.5rem 1.8rem !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05) !important;
    }

    /* 5. Horizontal Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(255, 255, 255, 0.65);
        padding: 8px 12px;
        border-radius: 9999px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px !important;
        line-height: 48px !important;
        border-radius: 9999px;
        padding: 0 22px !important;
        font-size: 1.02rem !important;
        font-weight: 600;
        color: #475569;
        background: transparent;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
    }

    /* 6. Metric Cards Readability */
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #334155 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        white-space: nowrap !important;
    }
    
    /* 7. Base Styling for ALL Streamlit Buttons */
    div.stButton > button {
        border-radius: 25px !important;
        padding: 0.7rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        width: 100% !important;
    }

    /* PRIMARY BUTTON: Royal Blue, Compact Height & Top Margin */
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"],
    .stButton button[kind="primary"] {
        background-color: #1E40AF !important;
        background: #1E40AF !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 0.45rem 1.5rem !important;
        min-height: 2.3rem !important;
        font-size: 0.95rem !important;
        margin-top: 0.8rem !important;
        box-shadow: 0 4px 14px rgba(30, 64, 175, 0.3) !important;
    }
    
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    .stButton button[kind="primary"]:hover {
        background-color: #1E3A8A !important;
        background: #1E3A8A !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4) !important;
        transform: translateY(-1px);
    }

    /* SECONDARY BUTTON: Crisp Text Contrast & Soft Pink Hover */
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"],
    .stButton button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.9) !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important;
        font-weight: 600 !important;
    }

    button[data-testid="baseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] p,
    .stButton button[kind="secondary"] p {
        color: #1E293B !important;
        font-weight: 600 !important;
    }

    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    .stButton button[kind="secondary"]:hover {
        background: #F4D5F0 !important;
        background-color: #F4D5F0 !important;
        border-color: rgba(255, 255, 255, 0.95) !important;
        box-shadow: 0 6px 20px rgba(244, 213, 240, 0.6) !important;
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# PAGE 1: CLASSIFIER WORKSPACE
# ----------------------------------------------------
if st.session_state.current_page == "classifier":
    st.markdown('<div class="project-badge-container"><span class="project-badge">✨ AI Medical Summary Intelligence</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Ask anything clinical.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Type clinical notes or trial summaries to classify target conditions instantly</div>', unsafe_allow_html=True)

    user_query = st.text_input("Clinical Search Query:", placeholder="+ Type clinical trial brief summary...", label_visibility="collapsed")

    # Column proportions [1.2, 1.6, 1.2] keep the primary button slim and centered
    col1, col2, col3 = st.columns([1.2, 1.6, 1.2])
    with col2:
        predict_clicked = st.button("🔍 Classify Clinical Note", type="primary", use_container_width=True)

    if (predict_clicked or user_query) and user_query.strip():
        cleaned_text = clean_clinical_text(user_query)
        
        with st.container(border=True):
            st.markdown("### 🩺 Classification Results")
            if not cleaned_text.strip():
                st.warning("⚠️ Input text contains no valid terms after preprocessing.")
            elif tfidf_vec is not None and lr_model is not None:
                try:
                    X_vec = tfidf_vec.transform([cleaned_text])
                    prediction = lr_model.predict(X_vec)[0]
                    if hasattr(lr_model, "predict_proba"):
                        probs = lr_model.predict_proba(X_vec)[0]
                        confidence = np.max(probs) * 100
                        if confidence < 50.0:
                            st.warning(f"**Low Confidence Prediction:** `{prediction}` ({confidence:.1f}% Confidence)\n\n💡 *Tip: Include more detail for better accuracy.*")
                        else:
                            st.success(f"**Target Condition Identified:** `{prediction}` ({confidence:.1f}% Confidence)")
                    else:
                        st.success(f"**Target Condition Identified:** `{prediction}`")
                except Exception as e:
                    st.error(f"Inference error: {e}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav2:
        st.button("📈 View Model Analysis & EDA Insights →", on_click=go_to_page, args=("analysis",), type='secondary', use_container_width=True)

# PAGE 2: MODEL ANALYSIS WITH SIDE-BY-SIDE TAB MENU
# ----------------------------------------------------
elif st.session_state.current_page == "analysis":
    st.button("← Back to Classifier", on_click=go_to_page, args=("classifier",))

    st.markdown('<div class="hero-title">EDA, Model Evaluation & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Comprehensive analytics as specified in the project requirements</div>', unsafe_allow_html=True)

    # Horizontal Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. Exploratory Data Analysis",
        "⚡ 2. Model Performance",
        "📈 3. Clinical Patterns",
        "🔍 4. Data Explorer"
    ])

    # TAB 1: EDA
    with tab1:
        with st.container(border=True):
            st.markdown("### 📊 Dataset Overview")
            
            # Key Dataset Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Studies", f"{len(df_data):,}")

            m2.metric(
                "Unique Conditions", 
                f"{df_data['source_condition_query'].nunique() if 'source_condition_query' in df_data.columns else 'N/A'}"
            )
            m3.metric(
                "Interventional", 
                f"{(df_data['study_type'] == 'INTERVENTIONAL').sum():,}" if 'study_type' in df_data.columns else "N/A",delta="77.5% of total studies"
            )
            m4.metric(
                "Completed Trials", 
                f"{(df_data['overall_status'] == 'COMPLETED').sum():,}" if 'overall_status' in df_data.columns else "N/A",delta="55.8% of total studies"
            )

            st.markdown("---")
            
            # Render Horizontal Viridis Chart
            render_interactive_distribution(df_data)

    # TAB 2: Model Performance
    with tab2:
        with st.container(border=True):
            st.markdown("### ⚡ Evaluation Metrics")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Accuracy", "94.85%")
            e2.metric("Precision", "95.00%")
            e3.metric("Recall", "95.00%")
            e4.metric("F1-Score", "95.00%")

            st.markdown("#### Model Architecture Telemetry")
            col_arch1, col_arch2 = st.columns(2)
            with col_arch1:
                vocab_dict = getattr(tfidf_vec, 'vocabulary_', None)
                st.json({
                    "Feature Extractor": "TF-IDF Vectorizer",
                    "Vocabulary Size": len(vocab_dict) if vocab_dict is not None else "20,000",
                    "N-gram Range": str(getattr(tfidf_vec, 'ngram_range', (1,3)))
                })
            with col_arch2:
                st.json({
                    "Model Algorithm": type(lr_model).__name__ if lr_model else "LogisticRegression",
                    "Classes Count": int(len(lr_model.classes_)) if hasattr(lr_model, 'classes_') else "8",
                    "Solver": getattr(lr_model, 'solver', 'sag')
                })
            st.markdown("---")
            st.markdown("#### 🎯 Confusion Matrix - Logistic Regression")

            # Plot interactively directly using saved array!
            fig_cm = px.imshow(
                cm,
                x=classes,
                y=classes,
                color_continuous_scale='Blues',
                text_auto=True,
                title="Confusion Matrix - Logistic Regression"
            )

            fig_cm.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=600,
                margin=dict(l=20, r=20, t=40, b=120)
            )
            fig_cm.update_xaxes(tickangle=-90)
            st.plotly_chart(fig_cm, use_container_width=True)

    # TAB 3: Clinical Patterns
    with tab3:
        with st.container(border=True):
            st.markdown("### 📈 Clinical Trial Metadata Patterns")
            c1, c2 = st.columns(2)
            with c1:
                if 'phase' in df_data.columns:
                    st.write("**Trial Phase Breakdown**")
                    st.bar_chart(df_data['phase'].value_counts().dropna())
            with c2:
                if 'sex' in df_data.columns:
                    st.write("**Sex Eligibility Breakdown**")
                    st.bar_chart(df_data['sex'].value_counts().dropna())

    # TAB 4: Data Explorer
    with tab4:
        with st.container(border=True):
            st.markdown("### 🔍 Interactive Data Search")
            
            f1, f2 = st.columns(2)
            with f1:
                query_filter = st.multiselect(
                    "Filter by Disease Category:",
                    options=df_data['source_condition_query'].dropna().unique() if 'source_condition_query' in df_data.columns else [],
                    default=[]
                )
            with f2:
                status_filter = st.multiselect(
                    "Filter by Trial Status:",
                    options=df_data['overall_status'].dropna().unique() if 'overall_status' in df_data.columns else [],
                    default=[]
                )

            filtered_df = df_data.copy()
            if query_filter:
                filtered_df = filtered_df[filtered_df['source_condition_query'].isin(query_filter)]
            if status_filter:
                filtered_df = filtered_df[filtered_df['overall_status'].isin(status_filter)]

            st.write(f"Displaying **{len(filtered_df):,}** records:")
            
            display_cols = ['nct_id', 'source_condition_query', 'title', 'overall_status', 'phase', 'clinicaltrials_url']
            existing_cols = [c for c in display_cols if c in filtered_df.columns]
            
            st.dataframe(
                filtered_df[existing_cols],
                column_config={"clinicaltrials_url": st.column_config.LinkColumn("ClinicalTrials Link")},
                use_container_width=True
            )