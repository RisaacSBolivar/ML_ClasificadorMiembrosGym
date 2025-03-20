from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(data, labels):
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    return silhouette, davies_bouldin