<div align="center">
  
  # 🌍 GeoImpact Risk Engine
  
  **An interactive Machine Learning dashboard simulating the economic fallout of geopolitical energy shocks on APAC supply chains.**

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
  [![XGBoost](https://img.shields.io/badge/XGBoost-172439.svg?style=for-the-badge&logo=XGBoost&logoColor=white)](https://xgboost.readthedocs.io/)
  [![Plotly](https://img.shields.io/badge/Plotly-3F4F75.svg?style=for-the-badge&logo=Plotly&logoColor=white)](https://plotly.com/)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KhannakPGupta/GeoImpact-Risk-Engine/blob/main/ensemble.ipynb)

</div>

---

## 📖 About The Project

What happens when global energy supply chains break? The **GeoImpact Risk Engine** is a predictive analytics platform designed to answer that question. 

By taking real-world constraints—such as a country's Strategic Petroleum Reserve (SPR) days, Middle East fuel dependency, and alternative sourcing capabilities—this engine runs mathematical simulations to forecast the cascading economic damage of regional conflicts. It serves as an interactive "what-if" tool for policymakers, analysts, and economists.

> **Live Demo:** *Deploy this app on Streamlit Cloud and link it here!*

---

## ✨ Key Features

- **📊 Dynamic Economic Forecasting:** Predicts the exact percentage surge in **Inflation** and **GDP Contraction**, alongside absolute **Import Cost** spikes.
- **🎛️ Interactive Scenario Engine:** Tweak shock factors (e.g., Oil Price Premium spikes up to 150%) to stress-test an economy's breaking point.
- **🧠 Machine Learning Core:** Powered by ensemble models (`XGBRegressor` & `XGBClassifier`) trained on synthetic geopolitical crisis data.
- **🔍 AI Explainability (SHAP):** Transparently breaks down *why* the model made its prediction, showing which supply-chain factors are driving inflation.
- **🌍 Country Vulnerability Clustering:** Uses `K-Means Clustering` to group APAC nations by their inherent supply chain fragilities.
- **🛡️ Actionable Policy Recommendations:** Generates immediate, data-driven legislative actions based on the severity of the predicted risk.

---

## 🛠️ Technology Stack

| Category | Technologies Used |
| :--- | :--- |
| **Frontend Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (K-Means, Preprocessing), XGBoost (Regression & Classification) |
| **Data Visualization** | Plotly Express, Plotly Graph Objects, Matplotlib |
| **AI Explainability** | SHAP (SHapley Additive exPlanations) |

---

## 🧠 The Machine Learning Pipeline

1. **Data Engineering:** Processes the `apac_fuel_import_dependency.csv` dataset, parsing categorical variables (`Fuel_Type`, `Conflict_Phase`) via `LabelEncoder`.
2. **Predictive Regression (XGBoost):** Three distinct `XGBRegressor` models forecast continuous economic metrics (Inflation Impact, GDP Impact, Import Cost Increase).
3. **Risk Classification (XGBoost):** An `XGBClassifier` categorizes the final situation into 4 discrete threat levels: *Low, Medium, High, Critical*.
4. **Unsupervised Clustering (K-Means):** Groups countries into 3 distinct vulnerability clusters based on their historical dependency and reserve levels using `StandardScaler` and distance metrics.

---

## 🚀 Getting Started (Local Setup)

To run this engine locally on your machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/KhannakPGupta/GeoImpact-Risk-Engine.git
cd GeoImpact-Risk-Engine
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
*The app will automatically open in your browser at `http://localhost:8501`.*

### (Optional) Retrain the Models
If you wish to re-train the XGBoost and K-Means models from scratch:
```bash
python ml_training.py
```
This will process the data, calculate new accuracy metrics, and overwrite the `.pkl` files in the `/models` directory.

---

## 📂 Project Structure

```text
GeoImpact-Risk-Engine/
├── app.py                            # Main Streamlit dashboard application
├── ml_training.py                    # Script to train and save ML models
├── apac_fuel_import_dependency.csv   # The core dataset
├── requirements.txt                  # Python package dependencies
├── ensemble.ipynb                    # Jupyter/Colab notebook for exploratory ML
└── models/                           # Directory containing pre-trained models
    ├── reg_inflation.pkl             # XGBoost model for Inflation
    ├── reg_gdp.pkl                   # XGBoost model for GDP
    ├── risk_classifier.pkl           # XGBoost model for Risk Level
    └── ...                           # Other encoders and scalers
```

---

<div align="center">
  <b>Built by <a href="https://github.com/KhannakPGupta">Khannak P Gupta</a></b>
</div>
