import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from utils.config import CLUSTER_MODEL_DIR

# Ensure directory exists
os.makedirs(CLUSTER_MODEL_DIR, exist_ok=True)

class StockClusterer:
    def __init__(self, symbols: list, start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        self.cluster_labels = None
        self.scaler = StandardScaler()
    
    def fetch_data(self):
        """Download historical data for multiple symbols"""
        closes = {}
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, start=self.start_date, end=self.end_date)
                closes[symbol] = df['Close']
            except Exception as e:
                print(f"Failed to download {symbol}: {str(e)}")
        
        self.data = pd.DataFrame(closes).dropna()
        return self.data
    
    def calculate_features(self):
        """Compute technical features for clustering"""
        features = pd.DataFrame(index=self.symbols)
        
        for symbol in self.symbols:
            if symbol not in self.data.columns:
                continue
                
            returns = self.data[symbol].pct_change().dropna()
            
            features.loc[symbol, 'mean_return'] = returns.mean()
            features.loc[symbol, 'volatility'] = returns.std()
            features.loc[symbol, 'sharpe_ratio'] = returns.mean() / returns.std()
            features.loc[symbol, 'max_drawdown'] = (self.data[symbol] / self.data[symbol].cummax() - 1).min()
            features.loc[symbol, 'skewness'] = stats.skew(returns)
            features.loc[symbol, 'kurtosis'] = stats.kurtosis(returns)
            features.loc[symbol, 'autocorrelation'] = returns.autocorr(lag=5)
        
        # Handle missing values
        self.features = features.dropna()
        return self.features
    
    def determine_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method"""
        scaled_features = self.scaler.fit_transform(self.features)
        distortions = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            distortions.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), distortions, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('Elbow Method for Optimal Clusters')
        plt.savefig(os.path.join(CLUSTER_MODEL_DIR, 'elbow_plot.png'))
        plt.close()
        
        return distortions
    
    def cluster_stocks(self, n_clusters=5, method='kmeans'):
        """Perform clustering on stock features"""
        scaled_features = self.scaler.fit_transform(self.features)
        
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=3)
        else:
            raise ValueError("Unsupported clustering method")
        
        self.cluster_labels = model.fit_predict(scaled_features)
        
        # Evaluate clustering
        if method == 'kmeans' and n_clusters > 1:
            silhouette = silhouette_score(scaled_features, self.cluster_labels)
            print(f"Silhouette Score: {silhouette:.3f}")
        
        # Save model and results
        joblib.dump(model, os.path.join(CLUSTER_MODEL_DIR, 'cluster_model.pkl'))
        self.save_cluster_assignments()
        
        return self.cluster_labels
    
    def save_cluster_assignments(self):
        """Save cluster assignments to CSV"""
        assignments = pd.DataFrame({
            'Symbol': self.features.index,
            'Cluster': self.cluster_labels
        })
        assignments.to_csv(os.path.join(CLUSTER_MODEL_DIR, 'cluster_assignments.csv'), index=False)
        return assignments
    
    def visualize_clusters(self, dim_reduction='pca'):
        """Visualize clusters using dimensionality reduction"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        scaled_features = self.scaler.fit_transform(self.features)
        
        if dim_reduction == 'pca':
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(scaled_features)
            x_label, y_label = 'PC1', 'PC2'
        else:
            reducer = TSNE(n_components=2, perplexity=5, random_state=42)
            reduced = reducer.fit_transform(scaled_features)
            x_label, y_label = 'TSNE1', 'TSNE2'
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced[:, 0], 
            reduced[:, 1], 
            c=self.cluster_labels, 
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        # Annotate points with symbols
        for i, symbol in enumerate(self.features.index):
            plt.annotate(symbol, (reduced[i, 0], reduced[i, 1]), fontsize=9)
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('Stock Clusters')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(CLUSTER_MODEL_DIR, 'cluster_visualization.png'))
        plt.close()
    
    def analyze_cluster_characteristics(self):
        """Analyze technical characteristics of each cluster"""
        cluster_data = self.features.copy()
        cluster_data['Cluster'] = self.cluster_labels
        
        # Group statistics
        cluster_stats = cluster_data.groupby('Cluster').agg(['mean', 'std'])
        
        # Visualize characteristics
        plt.figure(figsize=(14, 10))
        for i, feature in enumerate(self.features.columns):
            plt.subplot(3, 3, i+1)
            sns.boxplot(x='Cluster', y=feature, data=cluster_data)
            plt.title(feature)
            plt.tight_layout()
        
        plt.savefig(os.path.join(CLUSTER_MODEL_DIR, 'cluster_characteristics.png'))
        plt.close()
        
        return cluster_stats
    
    def get_cluster_peers(self, symbol: str):
        """Get stocks in the same cluster as a given symbol"""
        assignments = pd.read_csv(os.path.join(CLUSTER_MODEL_DIR, 'cluster_assignments.csv'))
        if symbol not in assignments['Symbol'].values:
            return []
        
        cluster = assignments[assignments['Symbol'] == symbol]['Cluster'].values[0]
        peers = assignments[assignments['Cluster'] == cluster]['Symbol'].tolist()
        peers.remove(symbol)  # Remove the target symbol
        return peers

# Example usage
if __name__ == "__main__":
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'TSLA', 'JPM', 'JNJ', 'XOM', 'WMT',
        'NVDA', 'V', 'PG', 'MA', 'DIS'
    ]
    
    clusterer = StockClusterer(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-01-01'
    )
    
    # Build clustering model
    clusterer.fetch_data()
    clusterer.calculate_features()
    distortions = clusterer.determine_optimal_clusters()
    clusterer.cluster_stocks(n_clusters=4)
    
    # Generate insights
    clusterer.visualize_clusters(dim_reduction='tsne')
    clusterer.analyze_cluster_characteristics()
    
    # Get peers for a stock
    print("Peers for AAPL:", clusterer.get_cluster_peers('AAPL'))