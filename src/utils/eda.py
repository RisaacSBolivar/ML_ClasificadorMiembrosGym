import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a given CSV filepath.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(filepath)
    return df

def explore_data(df: pd.DataFrame) -> None:
    """
    Performs basic Exploratory Data Analysis (EDA), printing stats
    and plotting distributions and a correlations heatmap.
    
    Args:
        df (pd.DataFrame): The input dataset.
    """
    print("Dataset Information / Información del dataset:")
    print(df.info())
    print("\nStatistical Description / Descripción estadística:")
    print(df.describe())
    print("\nNull values per column / Valores nulos por columna:")
    print(df.isnull().sum())
    
    numeric_cols = df.select_dtypes(include=[np.number])
    sns.set_theme(style="whitegrid")
    
    for col in numeric_cols.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(numeric_cols[col], bins=30, kde=True, color='teal')
        plt.title(f'Distribution of / Distribución de : {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency / Frecuencia')
        plt.tight_layout()
        plt.show()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap / Mapa de Calor de Correlación')
    plt.tight_layout()
    plt.show()