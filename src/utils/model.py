from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import joblib

def find_optimal_clusters(df: pd.DataFrame, k_range: range = range(2, 11)) -> None:
    """
    Calculates and plots the Elbow Method (Inertia) and Silhouette Score
    to find the optimal number of clusters for KMeans.
    
    Args:
        df (pd.DataFrame): The preprocessed dataset.
        k_range (range): The range of K values to evaluate.
    """
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(df)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))
    
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (K) / Número de clusters (K)')
    ax1.set_ylabel('Inertia / Inercia', color=color)
    ax1.plot(k_range, inertia, 'bo-', label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_scores, 'ro-', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Elbow Method and Silhouette Score / Método del Codo y Silhouette Score')
    plt.show()

def apply_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> np.ndarray:
    """
    Fits a KMeans model and returns the predicted cluster labels.
    
    Args:
        df (pd.DataFrame): The preprocessed data.
        n_clusters (int): The number of clusters to form.
        
    Returns:
        np.ndarray: Array of cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(df)
    return labels

def evaluate_clustering(df: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Evaluates the clustering quality using multiple metrics.
    
    Args:
        df (pd.DataFrame): The input dataset used for clustering.
        labels (np.ndarray): The labels predicted by the model.
    """
    silhouette = silhouette_score(df, labels)
    davies_bouldin = davies_bouldin_score(df, labels)
    calinski_harabasz = calinski_harabasz_score(df, labels)
    
    print(f'Silhouette Score: {silhouette:.4f}')
    print(f'Davies-Bouldin Score: {davies_bouldin:.4f}')
    print(f'Calinski-Harabasz Score: {calinski_harabasz:.4f}')

def find_optimal_eps(data: np.ndarray, k: int = 5) -> None:
    """
    Plots the K-distance graph to help define the optimal `eps` for DBSCAN.
    
    Args:
        data (np.ndarray): The scaled data.
        k (int): Number of neighbors to use for distance calculation.
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    distances = np.sort(distances[:, k-1])
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(distances, color='purple')
    plt.xlabel('Data points sorted by distance / Puntos de datos ordenados')
    plt.ylabel(f'Distance to {k}-th neighbor / Distancia al {k}-ésimo vecino')
    plt.title(f'Optimal eps curve / Curva para encontrar el valor óptimo de eps')
    plt.show()

def apply_dbscan(data: np.ndarray, eps_values: List[float], min_samples_values: List[int]) -> Tuple[float, int, np.ndarray]:
    """
    Performs grid search to find the best DBSCAN hyperparameters based on silhouette score.
    
    Args:
        data (np.ndarray): The scaled data.
        eps_values (List[float]): A list of `eps` values to evaluate.
        min_samples_values (List[int]): A list of `min_samples` values to evaluate.
        
    Returns:
        Tuple[float, int, np.ndarray]: Best eps, best min_samples, and the best cluster labels.
    """
    best_score = -1
    best_eps = None
    best_min_samples = None
    best_labels = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)
            
            # Silhouette score is only defined if number of clusters is 2 <= n_classes <= n_samples - 1
            if len(set(labels)) > 1 and len(set(labels)) < len(data):
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
    
    print(f'Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best Silhouette Score: {best_score:.4f}')
    return best_eps, best_min_samples, best_labels

def find_best_k(df_pca: np.ndarray, k_range: range = range(2, 11)) -> int:
    """
    Finds the optimal number of clusters for Spectral Clustering based on Silhouette Score.
    
    Args:
        df_pca (np.ndarray): Reduced dataset (PCA).
        k_range (range): Range of clusters to test.
        
    Returns:
        int: The optimal number of clusters (best K).
    """
    silhouette_scores = []
    sns.set_theme(style="whitegrid")
    
    for k in k_range:
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
        labels = model.fit_predict(df_pca)
        score = silhouette_score(df_pca, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--', color='teal')
    plt.xlabel('Number of Clusters / Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Selection of Best Clusters / Selección del Mejor Número de Clusters')
    plt.show()
    
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f'Best number of clusters / Mejor número de clusters: {best_k}')
    return best_k

def apply_spectral_clustering(df_pca: np.ndarray, best_k: int) -> np.ndarray:
    """
    Fits a Spectral Clustering model.
    
    Args:
        df_pca (np.ndarray): Reduced dataset (PCA).
        best_k (int): Number of clusters to form.
        
    Returns:
        np.ndarray: Array of cluster labels.
    """
    model = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
    labels = model.fit_predict(df_pca)
    return labels

def save_model(model: any, filename: str) -> None:
    """
    Saves a trained model to a file using joblib.
    
    Args:
        model (any): The trained scikit-learn model.
        filename (str): The destination filepath.
    """
    joblib.dump(model, filename)
    print(f'Model saved as / Modelo guardado como {filename}')

def evaluate_model(df_pca: np.ndarray, labels: np.ndarray) -> float:
    """
    Evaluates a clustering model's Final Silhouette Score.
    
    Args:
        df_pca (np.ndarray): Data used for standard clustering.
        labels (np.ndarray): Array of cluster labels.
        
    Returns:
        float: The silhouette score.
    """
    silhouette_avg = silhouette_score(df_pca, labels)
    print(f'Final Silhouette Score: {silhouette_avg:.4f}')
    return silhouette_avg