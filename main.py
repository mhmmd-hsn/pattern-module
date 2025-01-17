from feature_selection import FeatureSelection
from data_cleaning import DatasetCleaner    
import pandas as pd
import argparse
from visualization import Visualization
from dimention_reduction import DimensionalityReduction
from models import Classifiers

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(f'{args.dataset}.csv')
    # df = pd.read_csv(f'{args.data_path}.csv')
    df.rename(columns={ args.target : 'target'}, inplace=True)

    # ------------------------------------------------------------------------------------------------------------
    """Data Cleaning"""
    # takes care of Nan_values, outliers
    df = DatasetCleaner.remove_features_with_many_nans(df, threshold=0.5) 
    df = DatasetCleaner.handle_missing_values(df)
    # df = DatasetCleaner.iqr_outliers(df)
    # df = DatasetCleaner.dbscan_outliers(data=df, eps=0.5, min_samples=5, remove_outliers=True) 
    # DatasetCleaner.dataset_info(df)
    # ------------------------------------------------------------------------------------------------------------
    """Data Visualization""" 
    # viz = Visualization(df)
    # viz.plot_distributions() 
    # viz.plot_pairwise_relationships() 
    # viz.apply_tsne(n_components=3)
    # viz.plot_correlations()
    # viz.apply_pca(n_components=3)
    # viz.plot_distributions_by_label() 
    # viz.plot_feature_vs_target()
    # ------------------------------------------------------------------------------------------------------------#
    # fs = FeatureSelection(df)
    # fs.feature_selection_table()
    # --------------------------------------------------------------------------------------------------------#
    # """Dimension Reduction"""
    # dr = DimensionalityReduction(df)
    # # Apply PCA with optimal number of components (by variance threshold)
    # pca_result = dr.transform(method="PCA")
    # print("PCA Reduced Data Shape:", pca_result.head())    
    #---------------------------------------------------------------------------------------------------------#
    
    clf = Classifiers(df)
    # Train and evaluate all classifiers
    results = clf.train_all_classifiers()

    # Print evaluation metrics for each classifier
    for classifier, metrics in results.items():
        print(f"--- {classifier} ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("\n")
