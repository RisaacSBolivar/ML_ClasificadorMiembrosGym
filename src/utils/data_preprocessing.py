from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

def preprocess_data_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data for KMeans clustering by handling missing values, 
    encoding categorical variables, and scaling numerical features.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: The preprocessed dataframe ready for KMeans.
    """
    df = df.dropna().copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_cols = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = pd.concat([df[numeric_cols], encoded_df], axis=1)
    else:
        df = df[numeric_cols]
    
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled

def preprocess_data_dbscan(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocesses the data for DBSCAN clustering.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Original dataframe with encoded categorical 
        features, and the fully scaled numpy array for modeling.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, df_encoded], axis=1)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled

def reduce_dimensionality(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduces the dimensionality of the dataset using Principal Component Analysis (PCA).
    
    Args:
        data (np.ndarray): The input scaled data.
        n_components (int): The number of output components.
    
    Returns:
        np.ndarray: The PCA-reduced data.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

def preprocess_data_spectral(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocesses the data for Spectral Clustering.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        np.ndarray: The fully scaled array ready for Spectral Clustering.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, df_encoded], axis=1)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled
