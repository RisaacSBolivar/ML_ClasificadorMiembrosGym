import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import joblib
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
import numpy as np


def find_optimal_clusters(df, k_range=range(2, 11)):
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(df)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))
    
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(k_range, inertia, 'bo-', label='Inertia')
    ax1.set_xlabel('Número de clusters (K)')
    ax1.set_ylabel('Inertia', color='b')
    
    ax2 = ax1.twinx()
    ax2.plot(k_range, silhouette_scores, 'ro-', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color='r')
    
    plt.title('Método del Codo y Silhouette Score')
    plt.show()

def apply_kmeans(df, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(df)
    return labels

def evaluate_clustering(df, labels):
    silhouette = silhouette_score(df, labels)
    davies_bouldin = davies_bouldin_score(df, labels)
    calinski_harabasz = calinski_harabasz_score(df, labels)
    
    print(f'Silhouette Score: {silhouette:.4f}')
    print(f'Davies-Bouldin Score: {davies_bouldin:.4f}')
    print(f'Calinski-Harabasz Score: {calinski_harabasz:.4f}')


def find_optimal_eps(data, k=5):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    distances = np.sort(distances[:, k-1])
    plt.plot(distances)
    plt.xlabel('Puntos de datos ordenados')
    plt.ylabel(f'Distancia al {k}-ésimo vecino')
    plt.title('Curva para encontrar el valor óptimo de eps')
    plt.show()

def apply_dbscan(data, eps_values, min_samples_values):
    best_score = -1
    best_eps = None
    best_min_samples = None
    best_labels = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)
            
            if len(set(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
    
    print(f'Mejor eps: {best_eps}, Mejor min_samples: {best_min_samples}, Mejor Silhouette Score: {best_score}')
    return best_eps, best_min_samples, best_labels

def find_best_k(df_pca, k_range=range(2, 11)):
    silhouette_scores = []
    for k in k_range:
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')
        labels = model.fit_predict(df_pca)
        score = silhouette_score(df_pca, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Selección del Mejor Número de Clusters')
    plt.show()
    
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f'Mejor número de clusters: {best_k}')
    return best_k

def apply_spectral_clustering(df_pca, best_k):
    model = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', assign_labels='discretize')
    labels = model.fit_predict(df_pca)
    return labels

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f'Modelo guardado como {filename}')

def evaluate_model(df_pca, labels):
    silhouette_avg = silhouette_score(df_pca, labels)
    print(f'Valor final del Silhouette Score: {silhouette_avg}')
    return silhouette_avg