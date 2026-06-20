import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("--- Loading Dataset ---")
    try:
        df = pd.read_csv('apac_fuel_import_dependency.csv')
    except FileNotFoundError:
        print("Error: 'apac_fuel_import_dependency.csv' not found.")
        return

    print(f"\nDataset Shape: {df.shape}")
    
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Duplicates ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    print("\n--- Generating Visualizations ---")
    os.makedirs('eda_plots', exist_ok=True)
    
    # Set the style
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution of Numerical Features
    num_cols = ['Import_Volume_KBPD', 'ME_Share_Pct', 'Alternative_Source_Pct', 'Price_Premium_Pct', 'SPR_Days_Cover', 'Disruption_Risk_Score']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {col}')
        
    plt.tight_layout()
    plt.savefig('eda_plots/numerical_distributions.png')
    plt.close()
    
    # 2. Count Plots for Categorical Features
    cat_cols = ['Country', 'Fuel_Type', 'Conflict_Phase']
    
    for col in cat_cols:
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, x=col, hue=col, palette='viridis', legend=False)
            plt.title(f'Count Plot of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'eda_plots/{col}_countplot.png')
            plt.close()
        
    # 3. Correlation Matrix
    # We only compute correlation for numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig('eda_plots/correlation_matrix.png')
        plt.close()
    
    print("EDA completed! Visualizations saved in the 'eda_plots' directory.")

if __name__ == "__main__":
    run_eda()
