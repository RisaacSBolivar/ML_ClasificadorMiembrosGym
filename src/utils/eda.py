import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def explore_data(df):
    print("Información del dataset:")
    print(df.info())
    print("\nDescripción estadística:")
    print(df.describe())
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df.select_dtypes(include=[np.number]), bins=30, kde=True)
    plt.title('Distribución de las Variables Numéricas')
    plt.show()
    
    numeric_cols = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Calor de Correlación (Solo Variables Numéricas)')
    plt.show()