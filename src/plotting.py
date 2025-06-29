"""
This will handle the plotting of the data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_eda_plots():
    """Generate EDA plots for CHF dataset"""
    # Setup paths
    reports_dir = Path('reports/figures')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path('data/raw/heat_flux_data.csv')
    if not data_path.exists():
        print("⚠️ Data not found - run download first")
        return
    
    df = pd.read_csv(data_path)
    
    print("📊 Creating plots for CHF data...")
    
    # 1. Target Variable Distribution (chf_exp)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['chf_exp [MW/m2]'], kde=True, bins=20)
    plt.title('Critical Heat Flux (CHF) Distribution')
    plt.xlabel('CHF [MW/m2]')
    plt.ylabel('Frequency')
    plt.savefig(reports_dir / 'chf_distribution.png', bbox_inches='tight')
    plt.close()
    
    # 2. Pressure vs CHF
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pressure [MPa]', y='chf_exp [MW/m2]')
    plt.title('Pressure vs Critical Heat Flux')
    plt.savefig(reports_dir / 'pressure_vs_chf.png', bbox_inches='tight')
    plt.close()
    
    # 3. Mass Flux vs CHF
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mass_flux [kg/m2-s]', y='chf_exp [MW/m2]')
    plt.title('Mass Flux vs Critical Heat Flux')
    plt.savefig(reports_dir / 'massflux_vs_chf.png', bbox_inches='tight')
    plt.close()
    
    # 4. Correlation Heatmap (numerical only)
    numerical_cols = ['pressure [MPa]', 'mass_flux [kg/m2-s]', 
                     'x_e_out [-]', 'D_e [mm]', 'D_h [mm]', 
                     'length [mm]', 'chf_exp [MW/m2]']
    
    plt.figure(figsize=(12, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(reports_dir / 'correlation_matrix.png', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved 4 plots to {reports_dir}")
