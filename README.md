# 🌍 GeoImpact Risk Engine

**GeoImpact Risk Engine** is an interactive, AI-powered dashboard designed to simulate and analyze the economic shocks caused by geopolitical conflicts on APAC energy supply chains.

## 🚀 Features
- **Predictive Machine Learning**: Built with XGBoost and K-Means Clustering to forecast Inflation, GDP Contraction, and Import Cost Surges based on real-world constraints.
- **Interactive "What-If" Scenarios**: Adjust metrics like Import Volume, Middle East Dependency, and Strategic Petroleum Reserve (SPR) days to test economic breaking points.
- **Explainable AI**: Includes SHAP (SHapley Additive exPlanations) to transparently show which supply-chain factors are driving the predicted economic impact.
- **Instant Insights**: Actionable policy recommendations based on the simulated risk level.

## 🛠️ Tech Stack
- **Frontend**: Streamlit, Plotly
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Processing**: Pandas, NumPy

## 🏃‍♂️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/KhannakPGupta/GeoImpact-Risk-Engine.git
   cd GeoImpact-Risk-Engine
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 🧠 Machine Learning Training
The models are pre-trained and stored in the `models/` directory, but you can retrain the engine from scratch by running:
```bash
python ml_training.py
```
This script processes the `apac_fuel_import_dependency.csv` dataset, trains XGBoost Regressors for economic impact, an XGBoost Classifier for Risk Level, and applies K-Means Clustering to identify vulnerable countries.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KhannakPGupta/GeoImpact-Risk-Engine/blob/main/ensemble.ipynb) *(Make sure to upload your notebook to this repo!)*
