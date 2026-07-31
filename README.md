# clinical-trial-summary-prediction

# 🩺 AI Medical Summary Intelligence & Clinical Trial Classifier

An end-to-end Machine Learning and NLP web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** that automatically classifies clinical trial brief summaries and medical notes into target condition categories in real time.

---

## 📸 Key Features

- **⚡ Real-Time Classification:** Input custom clinical notes or trial briefs to receive instant model predictions.
- **📊 Model Performance Analytics:** Interactive evaluation telemetry including accuracy metrics, classification reports, and model comparisons.
- **🎯 Dynamic Confusion Matrix:** Interactive Plotly heatmap visualization powered by pre-calculated evaluation artifacts for ultra-fast app rendering.
- **🎨 Modern UI/UX:** Built with a custom glassmorphism design, styled primary/secondary call-to-action buttons, and clean layout responsive formatting.

---

## 📁 Repository Structure

```text
├── app.py                         # Main Streamlit web application
├── clinical_trial_classifier.pkl  # Trained Logistic Regression classifier artifact
├── tfidf_vectorizer.pkl           # Fitted TF-IDF Vectorizer artifact
├── confusion_matrix.pkl           # Pre-calculated Confusion Matrix array artifact
├── model_classes.pkl              # Target class labels artifact
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
