import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

def plot_clusters_kmeans(data, labels, title="Clustering Visualization"):
    """Genera un scatter plot con los clusters encontrados."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_cluster_distribution(labels, title="Distribución de Clusters"):
    """Genera un gráfico de barras con la cantidad de elementos en cada cluster."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=unique_labels, y=counts, palette="viridis")
    plt.xlabel("Cluster")
    plt.ylabel("Número de Elementos")
    plt.title(title)
    plt.show()

def plot_clusters_spectral(df_pca, labels):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=labels, palette='viridis')
    plt.title('Spectral Clustering')
    plt.show()


def visualize_clusters_dbscan(data, labels):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis')
    plt.title('DBSCAN Clustering')
    plt.show()

def visualize_clusters_kmeans(df, labels):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], hue=labels, palette='viridis', alpha=0.7)
    plt.title('Clusters visualizados con PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()