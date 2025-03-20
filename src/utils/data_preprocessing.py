import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

def preprocess_data_kmeans(df):
    df = df.dropna()
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_cols = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df[numeric_cols], encoded_df], axis=1)
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess_data_dbscan(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, df_encoded], axis=1)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled

def reduce_dimensionality(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

def preprocess_data_spectral(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, df_encoded], axis=1)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled
