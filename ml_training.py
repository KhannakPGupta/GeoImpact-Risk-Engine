import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os

def create_synthetic_targets(df):
    """Generate realistic economic impact targets based on existing features with real-world noise."""
    np.random.seed(42) # For reproducibility
    
    # Base formulas
    base_import_cost = df['Price_Premium_Pct'] * (df['ME_Share_Pct'] / 100) * 1.2
    base_inflation = (base_import_cost * 0.15) + (df['Import_Volume_KBPD'] / 20000)
    base_gdp = -(base_inflation * 0.8) - (50 / (df['SPR_Days_Cover'] + 1))
    
    # Add Gaussian Noise (Real-world unpredictability)
    # The noise scales with the magnitude of the value to keep it realistic
    df['Import_Cost_Increase'] = base_import_cost + np.random.normal(0, base_import_cost.mean() * 0.2, len(df))
    df['Inflation_Impact'] = base_inflation + np.random.normal(0, base_inflation.mean() * 0.2, len(df))
    df['GDP_Impact'] = base_gdp + np.random.normal(0, abs(base_gdp.mean()) * 0.2, len(df))
    
    # Ensure no negative import costs or inflation (edge cases from noise)
    df['Import_Cost_Increase'] = df['Import_Cost_Increase'].clip(lower=0)
    df['Inflation_Impact'] = df['Inflation_Impact'].clip(lower=0)
    df['GDP_Impact'] = df['GDP_Impact'].clip(upper=0)
    
    # Risk Classification (0: Low, 1: Medium, 2: High, 3: Critical)
    conditions = [
        (df['Disruption_Risk_Score'] > 2.5) & (df['SPR_Days_Cover'] < 90),
        (df['Disruption_Risk_Score'] > 1.5),
        (df['Disruption_Risk_Score'] > 0.5)
    ]
    choices = [3, 2, 1]
    
    # Add occasional misclassifications to simulate real-world ambiguity in risk assessment
    base_risk = np.select(conditions, choices, default=0)
    noise_mask = np.random.rand(len(base_risk)) < 0.05 # 5% of data has shifted risk
    df['Risk_Class'] = np.where(noise_mask, np.clip(base_risk + np.random.choice([-1, 1], len(base_risk)), 0, 3), base_risk)
    
    return df

def train_models():
    print("Loading data...")
    df = pd.read_csv('apac_fuel_import_dependency.csv')
    
    print("Feature Engineering & Adding Noise...")
    df = create_synthetic_targets(df)
    
    # Encode Categorical Variables
    encoders = {}
    cat_cols = ['Country', 'Fuel_Type', 'Conflict_Phase']
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_Encoded'] = le.fit_transform(df[col])
        encoders[col] = le
        
    features = [
        'Country_Encoded', 'Fuel_Type_Encoded', 'Conflict_Phase_Encoded',
        'Import_Volume_KBPD', 'ME_Share_Pct', 'Alternative_Source_Pct', 
        'Price_Premium_Pct', 'SPR_Days_Cover'
    ]
    
    X = df[features]
    
    print("\n--- Model Evaluation (Train/Test Split) ---")
    
    # Helper to train, evaluate, and return regression models
    def train_and_eval_regressor(target_name, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        
        # Fit on training data ONLY
        model.fit(X_train, y_train)
        
        # Evaluate on unseen testing data
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        print(f"[{target_name}] R2 Score: {r2:.4f} | RMSE: {rmse:.4f}")
        return model

    # 1. Train Economic Impact Predictors
    reg_inflation = train_and_eval_regressor("Inflation Impact", df['Inflation_Impact'])
    reg_gdp = train_and_eval_regressor("GDP Impact", df['GDP_Impact'])
    reg_import = train_and_eval_regressor("Import Cost Increase", df['Import_Cost_Increase'])
    
    # 2. Train Risk Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, df['Risk_Class'], test_size=0.2, random_state=42)
    classifier = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    classifier.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, classifier.predict(X_test))
    print(f"[Risk Classification] Accuracy: {acc:.4f} (Not 100% due to added noise)")
    
    # 3. Country Vulnerability Analysis (K-Means)
    print("\nTraining K-Means...")
    country_stats = df.groupby('Country').agg({
        'ME_Share_Pct': 'mean',
        'SPR_Days_Cover': 'mean',
        'Disruption_Risk_Score': 'mean'
    }).reset_index()
    
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(country_stats[['ME_Share_Pct', 'SPR_Days_Cover', 'Disruption_Risk_Score']])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    country_stats['Cluster'] = kmeans.fit_predict(scaled_stats)
    
    # Save everything
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(reg_inflation, 'models/reg_inflation.pkl')
    joblib.dump(reg_gdp, 'models/reg_gdp.pkl')
    joblib.dump(reg_import, 'models/reg_import.pkl')
    joblib.dump(classifier, 'models/risk_classifier.pkl')
    joblib.dump(kmeans, 'models/kmeans_country.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(scaler, 'models/kmeans_scaler.pkl')
    country_stats.to_csv('models/country_clusters.csv', index=False)
    
    print("Training Complete! Validated models saved to /models directory.")

if __name__ == "__main__":
    train_models()
