import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def plot_clusters_kmeans(data: np.ndarray, labels: np.ndarray, title: str = "K-Means Clustering Visualization") -> None:
    """
    Generates a scatter plot of the found K-Means clusters.
    
    Args:
        data (np.ndarray): The 2D array data to plot.
        labels (np.ndarray): The predicted cluster labels.
        title (str): Title of the plot.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.8, edgecolors='k')
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("Principal Component 1 / Componente Principal 1")
    plt.ylabel("Principal Component 2 / Componente Principal 2")
    plt.show()

def plot_cluster_distribution(labels: np.ndarray, title: str = "Cluster Distribution / Distribución de Clusters") -> None:
    """
    Generates a bar plot showing the quantity of items in each cluster.
    
    Args:
        labels (np.ndarray): Array of cluster labels.
        title (str): Title of the plot.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    
    # Ensuring qualitative palette and plotting
    sns.barplot(x=unique_labels, y=counts, palette="viridis", hue=unique_labels, legend=False)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Elements / Número de Elementos")
    plt.title(title)
    plt.show()

def plot_clusters_spectral(df_pca: np.ndarray, labels: np.ndarray) -> None:
    """
    Generates a scatter plot of the Spectral Clustering.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=labels, palette='viridis', alpha=0.8, edgecolor='k')
    plt.title('Spectral Clustering Results')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.show()

def visualize_clusters_dbscan(data: np.ndarray, labels: np.ndarray) -> None:
    """
    Visualizes the DBSCAN clusters using a 2D scatter plot.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Separate noise (-1) to plot in gray/black
    unique_labels = np.unique(labels)
    palette = sns.color_palette("viridis", len(unique_labels) - (1 if -1 in labels else 0))
    color_dict = {lbl: palette[i] for i, lbl in enumerate(lbl for lbl in unique_labels if lbl != -1)}
    if -1 in labels:
        color_dict[-1] = (0.3, 0.3, 0.3) # Dark gray for noise
        
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=color_dict, alpha=0.8, edgecolor='k')
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Cluster')
    plt.show()

def visualize_clusters_kmeans(df: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Applies PCA internally to visualize K-Means clusters on a high-dimensional dataframe.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='viridis', alpha=0.8, edgecolor='k')
    plt.title('K-Means Clusters (PCA-Reduced) / Clusters visualizados con PCA')
    plt.xlabel('Principal Component 1 / Componente Principal 1')
    plt.ylabel('Principal Component 2 / Componente Principal 2')
    plt.legend(title='Cluster')
    plt.show()