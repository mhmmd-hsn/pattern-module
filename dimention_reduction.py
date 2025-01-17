import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from data_cleaning import DatasetCleaner

class DimensionalityReduction:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DimensionalityReduction object.
        
        Parameters:
        data (pd.DataFrame): The dataset to perform dimension reduction on.
        """
        self.data = data
        self.scaled_data, _ = DatasetCleaner.standardization(self.data.drop("target", axis=1))
        self.pca_model = None
        self.tsne_model = None

    def find_optimal_pca_components(self, variance_threshold=0.95):
        """
        Finds the optimal number of components for PCA that retains a certain amount of variance.
        
        Parameters:
        variance_threshold (float): The cumulative explained variance ratio to retain. Default is 0.95 (95%).
        
        Returns:
        int: The number of components that retains the specified variance.
        """
        pca = PCA()
        pca.fit(self.scaled_data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        return optimal_components

    def apply_pca(self, n_components=None):
        """
        Applies PCA to the data, optionally reducing to n_components.
        
        Parameters:
        n_components (int, optional): The number of components to keep. If None, will use optimal components based on variance.
        
        Returns:
        np.ndarray: The transformed data.
        """
        if n_components is None:
            n_components = self.find_optimal_pca_components()
        
        pca = PCA(n_components=n_components)
        self.pca_model = pca
        pca_result = pca.fit_transform(self.scaled_data)
        return pca_result

    def apply_tsne(self, perplexity_range=(5, 50), n_iter=1000, random_state=42):
        """
        Applies t-SNE to the data to reduce dimensionality.
        
        Parameters:
        perplexity_range (tuple): The range of perplexity values to test.
        n_iter (int): The number of iterations for the t-SNE optimization.
        random_state (int): The random seed for reproducibility.
        
        Returns:
        np.ndarray: The transformed data from t-SNE.
        """
        best_perplexity = None
        best_score = float('inf')
        best_tsne_result = None

        for perplexity in range(perplexity_range[0], perplexity_range[1] + 1, 5):
            tsne = TSNE(perplexity=perplexity, n_iter=n_iter, random_state=random_state)
            tsne_result = tsne.fit_transform(self.scaled_data)
            
            # Use pairwise distance to measure preservation of original distances (optional evaluation method)
            distance = pairwise_distances_argmin_min(tsne_result, self.scaled_data)[1].mean()

            if distance < best_score:
                best_score = distance
                best_perplexity = perplexity
                best_tsne_result = tsne_result

        print(f"Optimal Perplexity for t-SNE: {best_perplexity}")
        self.tsne_model = TSNE(perplexity=best_perplexity, n_iter=n_iter, random_state=random_state)
        return best_tsne_result

    def transform(self, method="PCA", n_components=None, perplexity_range=(5, 50), n_iter=1000):
        """
        Transforms the data using the specified dimensionality reduction method.
        
        Parameters:
        method (str): The method to use ('PCA' or 'tSNE').
        n_components (int, optional): The number of components for PCA. Default is None (use optimal PCA).
        perplexity_range (tuple): The range of perplexity values for t-SNE.
        n_iter (int): The number of iterations for t-SNE.
        
        Returns:
        np.ndarray: The transformed data.
        """  

        if method == "PCA":
            return self.apply_pca(n_components)
        elif method == "tSNE":
            return self.apply_tsne(perplexity_range, n_iter)
        else:
            raise ValueError("Invalid method. Choose 'PCA' or 'tSNE'.")

# Usage Example
if __name__ == "__main__":
    # Example dataset (replace with your actual dataset)
    data = pd.read_csv("diabetes.csv")

    # Initialize the DimensionalityReduction object
    dr = DimensionalityReduction(data)

    # Apply PCA with optimal number of components (by variance threshold)
    pca_result = dr.transform(method="PCA")
    print("PCA Reduced Data Shape:", pca_result.shape)

    # Apply t-SNE
    tsne_result = dr.transform(method="tSNE")
    print("t-SNE Reduced Data Shape:", tsne_result.shape)
