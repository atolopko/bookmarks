import os
from sentence_transformers import SentenceTransformer
import sys
import csv
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP

# List of models to try (without prefix)
MODELS = [
    'all-mpnet-base-v2',
    'multi-qa-mpnet-base-dot-v1',
    'all-distilroberta-v1',
    'paraphrase-multilingual-mpnet-base-v2',
    'multi-qa-MiniLM-L6-cos-v1',
    'all-MiniLM-L12-v2'
]

def get_full_model_name(model_suffix):
    return f"sentence-transformers/{model_suffix}"

def print_available_models():
    print("\nAvailable models:")
    for model in MODELS:
        print(model)

def read_bookmarks(csv_path, limit=None):
    with open(csv_path, "r") as file:
        csv_reader = csv.DictReader(file)
        bookmarks = [{k:v for k,v in row.items() if k != 'created_at'} for row in csv_reader]
    
    if limit:
        bookmarks = bookmarks[0:limit]
    
    header = list(bookmarks[0].keys())
    print(f"Header: {header}")
    print(f"First 3 rows: {bookmarks[:3]}")
    
    return bookmarks

def compute_embeddings(bookmarks, model_suffix='all-MiniLM-L6-v2', device="mps"):
    embeddings_file = f'embeddings_{model_suffix}.npy'
    
    if os.path.exists(embeddings_file):
        print(f"Loading existing embeddings from {embeddings_file}")
        return np.load(embeddings_file)
    
    print(f"Computing embeddings using model: {model_suffix}")
    model = SentenceTransformer(get_full_model_name(model_suffix))
    embeddings = model.encode(bookmarks, show_progress_bar=True, 
                            output_value="sentence_embedding",
                            device=device)
    
    np.save(embeddings_file, embeddings)
    print(f"Saved embeddings to {embeddings_file} with shape {embeddings.shape}")
    
    return embeddings

def generate_visualization(embeddings, bookmarks, model_suffix):
    output_file = f'embeddings_visualization_{model_suffix}.html'
    
    # Reduce dimensionality with UMAP
    reducer = UMAP(n_components=2, random_state=43)
    embedding_2d = reducer.fit_transform(embeddings)
    # Perform clustering on the 2D embeddings
    # Scale the data first for better clustering
    scaler = StandardScaler()
    embedding_2d_scaled = scaler.fit_transform(embedding_2d)
    
    # Apply KMeans clustering on full embeddings
    n_clusters = 10  # You may want to tune this parameter
    clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    labels = clustering.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"\nClustering results for model {model_suffix}:")
    print(f"Number of clusters found: {n_clusters}")
    
    # Count bookmarks in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"Noise points: {count}")
        else:
            print(f"Cluster {label}: {count} bookmarks")
    
    # Create interactive scatter plot with Plotly
    hover_text = [f"Title: {b.get('title', 'N/A')}<br>URL: {b.get('url', 'N/A')}<br>Cluster: {label}" 
                 for b, label in zip(bookmarks, labels)]
    
    # Create a color scale that handles noise points (-1) differently
    colors = [f'Cluster {label}' if label != -1 else 'Noise' for label in labels]
    
    fig = px.scatter(
        x=embedding_2d_scaled[:, 0],
        y=embedding_2d_scaled[:, 1],
        color=colors,
        hover_name=hover_text,
        title=f'Bookmark Embeddings Clusters using {model_suffix}',
        width=1200,
        height=800
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        hovermode='closest',
        showlegend=True
    )
    
    fig.write_html(output_file)
    print(f"Interactive visualization saved as {output_file}")
    
    return labels

def process_with_model(bookmarks, model_suffix):
    print(f"\nProcessing with model: {model_suffix}")
    embeddings = compute_embeddings(bookmarks, model_suffix=model_suffix)
    cluster_labels = generate_visualization(embeddings, bookmarks, model_suffix)
    return embeddings, cluster_labels

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embed.py <csv_file> [model_name]")
        print_available_models()
        sys.exit(1)
    
    bookmarks = read_bookmarks(sys.argv[1], limit=None)
    
    # If model name is provided, use that model, otherwise use the first one
    if len(sys.argv) > 2:
        model_suffix = sys.argv[2]
        if model_suffix not in MODELS:
            print(f"Error: Unknown model '{model_suffix}'")
            print_available_models()
            sys.exit(0)
    else:
        model_suffix = MODELS[0]  # default to first model
    
    embeddings, cluster_labels = process_with_model(bookmarks, model_suffix)
    print(embeddings.shape)


