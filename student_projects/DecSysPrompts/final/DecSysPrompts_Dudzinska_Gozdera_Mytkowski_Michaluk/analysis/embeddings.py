"""Embedding and interactive visualization for prompt-space.

This script builds embeddings for prompts, reduces dimensionality,
performs hierarchical clustering, and produces interactive 2D/3D
visualizations and a dendrogram to represent the "family tree"
of system prompts.
"""
import umap
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import spatial
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from config import SBERT_MODEL_NAME
from utils import load_prompts

def reduce_dimensions(embeddings, n_components=2, method='umap'):
    """Reduce dimensionality using UMAP, PCA or TSNE."""
    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)
    if method == 'pca':
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(embeddings)

    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(embeddings)

def find_optimal_clusters(embeddings, max_clusters=10):
    """Find optimal number of clusters using elbow method."""
    inertias = []
    cluster_range = range(2, min(max_clusters + 1, len(embeddings)))
    
    for n in cluster_range:
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
    
    angles = []
    for i in range(1, len(inertias) - 1):
        v1 = np.array([1, inertias[i] - inertias[i-1]])
        v2 = np.array([1, inertias[i+1] - inertias[i]])
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    
    optimal_idx = np.argmax(angles) + 1
    optimal_clusters = list(cluster_range)[optimal_idx]
    
    print(f"Elbow method suggests {optimal_clusters} clusters")
    return optimal_clusters

def cluster_embeddings(embeddings, n_clusters=None, method='kmeans', max_clusters=10):
    """Cluster embeddings using specified method."""

    if n_clusters is None and method in ['kmeans', 'agglomerative', 'spectral']:
        n_clusters = find_optimal_clusters(embeddings, max_clusters=max_clusters)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return clusterer.fit_predict(embeddings)
    elif method == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        return clusterer.fit_predict(embeddings)
    elif method == 'spectral':
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
        return clusterer.fit_predict(embeddings)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=2)
        return clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

def plot_interactive_2d(df, emb2d, color='provider', clusters=None):
    """Plot interactive 2D scatter"""
    plot_df = df.copy()
    plot_df['x'] = emb2d[:, 0]
    plot_df['y'] = emb2d[:, 1]
    if clusters is not None:
        plot_df['cluster'] = clusters.astype(str)

    hover_data = ['model', 'version', 'path', 'release_date']
    if clusters is not None:
        hover_data.append('cluster')

    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color=color,
        hover_data=hover_data,
        #title='Prompt-space (2D)'
    )
    
    if clusters is not None:
        cluster_colors = px.colors.qualitative.Plotly
        
        for cluster_id in sorted(set(clusters)):
            mask = clusters == cluster_id
            cluster_points = emb2d[mask]
            
            # Try to draw convex hull if there are enough points
            if len(cluster_points) >= 3:
                try:
                    hull = spatial.ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    
                    color_idx = cluster_id % len(cluster_colors)
                    cluster_color = cluster_colors[color_idx]
                    
                    fig.add_trace(go.Scatter(
                        x=hull_points[:, 0],
                        y=hull_points[:, 1],
                        fill='toself',
                        fillcolor=cluster_color,
                        opacity=0.2,
                        line=dict(color=cluster_color, width=2),
                        name=f'Cluster {cluster_id}',
                        hovertemplate=f'Cluster {cluster_id} Border<extra></extra>',
                        showlegend=True
                    ))
                except Exception as e:
                    print(f"Could not compute convex hull for cluster {cluster_id}: {e}")
            else:
                # For small clusters, just draw a circle
                color_idx = cluster_id % len(cluster_colors)
                cluster_color = cluster_colors[color_idx]
                center = cluster_points.mean(axis=0)
                radius = 0.1
                theta = np.linspace(0, 2*np.pi, 50)
                x_circle = center[0] + radius * np.cos(theta)
                y_circle = center[1] + radius * np.sin(theta)

                fig.add_trace(go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    fill='toself',
                    fillcolor=cluster_color,
                    opacity=0.2,
                    line=dict(color=cluster_color, width=2),
                    name=f'Cluster {cluster_id}',
                    hovertemplate=f'Cluster {cluster_id} Border<extra></extra>',
                    showlegend=True
                ))

    fig.update_traces(marker=dict(size=18), selector=dict(mode='markers'))
    fig.update_layout(legend=dict(font=dict(size=28)))
    fig.show()


def plot_interactive_3d(df, emb3d, color='provider', clusters=None):
    """Plot interactive 3D scatter"""
    plot_df = df.copy()
    plot_df['x'] = emb3d[:, 0]
    plot_df['y'] = emb3d[:, 1]
    plot_df['z'] = emb3d[:, 2]
    if clusters is not None:
        plot_df['cluster'] = clusters.astype(str)

    hover_data = ['model', 'version', 'path', 'release_date']
    if clusters is not None:
        hover_data.append('cluster')

    fig = px.scatter_3d(
        plot_df,
        x='x', y='y', z='z',
        color=color,
        hover_data=hover_data,
        #title='Prompt-space (3D)'
    )

    if clusters is not None:
        cluster_colors = px.colors.qualitative.Plotly
        
        for cluster_id in sorted(set(clusters)):
            mask = clusters == cluster_id
            cluster_points = emb3d[mask]
            
            color_idx = cluster_id % len(cluster_colors)
            cluster_color = cluster_colors[color_idx]
            
            # Try to draw convex hull if there are enough points
            if len(cluster_points) >= 4:
                try:
                    hull = spatial.ConvexHull(cluster_points)
                    
                    hull_vertices = cluster_points[hull.vertices]
                    
                    fig.add_trace(go.Mesh3d(
                        x=hull_vertices[:, 0],
                        y=hull_vertices[:, 1],
                        z=hull_vertices[:, 2],
                        i=hull.simplices[:, 0],
                        j=hull.simplices[:, 1],
                        k=hull.simplices[:, 2],
                        opacity=0.2,
                        color=cluster_color,
                        showlegend=True,
                        name=f'Cluster {cluster_id}',
                        hovertemplate=f'Cluster {cluster_id}<extra></extra>'
                    ))
                except Exception as e:
                    print(f"Could not compute convex hull for cluster {cluster_id}: {e}")
                    # Fallback: add cluster center marker
                    center = cluster_points.mean(axis=0)
                    fig.add_trace(go.Scatter3d(
                        x=[center[0]], y=[center[1]], z=[center[2]],
                        mode='markers',
                        marker=dict(size=12, color=cluster_color, symbol='diamond', 
                                    line=dict(width=2, color='darkred')),
                        name=f'Cluster {cluster_id}',
                        hovertemplate=f'Cluster {cluster_id} Center<extra></extra>',
                        showlegend=True
                    ))
            else:
                center = cluster_points.mean(axis=0)
                radius = np.std(cluster_points) if len(cluster_points) > 1 else 0.1
                
                fig.add_trace(go.Scatter3d(
                    x=[center[0]], y=[center[1]], z=[center[2]],
                    mode='markers',
                    marker=dict(size=15, color=cluster_color, symbol='diamond',
                                line=dict(width=2, color='darkred')),
                    name=f'Cluster {cluster_id}',
                    hovertemplate=f'Cluster {cluster_id} Center<extra></extra>',
                    showlegend=True
                ))
                
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                fig.add_trace(go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    opacity=0.15,
                    colorscale=[[0, cluster_color], [1, cluster_color]],
                    showscale=False,
                    showlegend=False,
                    hovertemplate=f'Cluster {cluster_id}<extra></extra>'
                ))
    
    fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
    fig.update_layout(legend=dict(font=dict(size=24)))
    fig.show()


def plot_dendrogram(embeddings, labels, save_path=None):
    """Plot a dendrogram for the hierarchical clustering."""
    plt.figure(figsize=(10,4))
    Z = linkage(embeddings, method='ward')
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=14)
    #plt.title('Hierarchical clustering dendrogram (family tree)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
def embeddings_analysis(save_path=None):
    """Main function to perform embeddings analysis on prompts."""
    
    df = load_prompts(use_cleaned=True)
    texts = df['text'].fillna('').tolist()

    print(f"Computing embeddings for {len(texts)} prompts...")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    n_providers = df['provider'].nunique()
    print(f"Found {n_providers} providers. Computing clusters...")
    clusters = cluster_embeddings(embeddings, n_clusters=None, method='spectral')

    emb2d = reduce_dimensions(embeddings, n_components=2, method='umap')
    plot_interactive_2d(df, emb2d, color='provider', clusters=clusters)

    if embeddings.shape[1] >= 3:
        emb3d = reduce_dimensions(embeddings, n_components=3, method='umap')
        plot_interactive_3d(df, emb3d, color='provider', clusters=None)

    print("Computing hierarchical clustering for family tree...")

    #labels = [f"{r['provider']}/{r['model']}" for _, r in df.iterrows()]
    labels = [f"{r['provider']}" for _, r in df.iterrows()]
    plot_dendrogram(embeddings, labels, save_path=save_path)

if __name__ == "__main__":
    embeddings_analysis()
