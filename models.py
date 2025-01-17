import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_cleaning import DatasetCleaner
class Classifiers:
    def __init__(self, df, test_size=0.2, random_state=42):
        """
        Initializes the Classifiers object.

        Parameters:
        data (pd.DataFrame): The dataset to train and test classifiers on.
        target_column (str): The name of the target column in the dataset.
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.3.
        random_state (int): The random seed for reproducibility. Default is 42.
        """
        self.X = df.drop("target", axis=1)
        self.y = df["target"]
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        self.scaled_X_train, self.scaled_X_test = DatasetCleaner.standardization(self.X_train, self.X_test)

    def evaluate(self, model, y_pred):
        """
        Evaluates the model performance.

        Parameters:
        model: The trained model.
        y_pred: Predictions from the model.
        
        Returns:
        dict: A dictionary of performance metrics (accuracy, precision, recall, F1 score).
        """
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

    def train_svm(self, kernel='linear', C=1.0):
        """
        Trains a Support Vector Machine (SVM) classifier.
        
        Parameters:
        kernel (str): The kernel to use. Can be 'linear' or 'rbf' (default is 'linear').
        C (float): Regularization parameter (default is 1.0).
        
        Returns:
        dict: Evaluation metrics after training.
        """
        svm = SVC(kernel=kernel, C=C, random_state=self.random_state)
        svm.fit(self.scaled_X_train, self.y_train)
        y_pred = svm.predict(self.scaled_X_test)
        return self.evaluate(svm, y_pred)

    def train_knn(self, n_neighbors=5, metric='minkowski'):
        """
        Trains a K-Nearest Neighbors (KNN) classifier.
        
        Parameters:
        n_neighbors (int): Number of neighbors to use (default is 5).
        metric (str): Distance metric to use (default is 'minkowski').
        
        Returns:
        dict: Evaluation metrics after training.
        """
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        knn.fit(self.scaled_X_train, self.y_train)
        y_pred = knn.predict(self.scaled_X_test)
        return self.evaluate(knn, y_pred)

    def train_random_forest(self, n_estimators=100, max_depth=None):
        """
        Trains a Random Forest classifier.
        
        Parameters:
        n_estimators (int): The number of trees in the forest (default is 100).
        max_depth (int): The maximum depth of the tree (default is None, meaning nodes are expanded until they contain less than min_samples_split samples).
        
        Returns:
        dict: Evaluation metrics after training.
        """
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_state)
        rf.fit(self.X_train, self.y_train)
        y_pred = rf.predict(self.X_test)
        return self.evaluate(rf, y_pred)

    def train_decision_tree(self, max_depth=None):
        """
        Trains a Decision Tree classifier.
        
        Parameters:
        max_depth (int): The maximum depth of the tree (default is None).
        
        Returns:
        dict: Evaluation metrics after training.
        """
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_state)
        dt.fit(self.X_train, self.y_train)
        y_pred = dt.predict(self.X_test)
        return self.evaluate(dt, y_pred)

    def train_naive_bayes(self):
        """
        Trains a Naive Bayes classifier (Gaussian Naive Bayes).
        
        Returns:
        dict: Evaluation metrics after training.
        """
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        y_pred = nb.predict(self.X_test)
        return self.evaluate(nb, y_pred)

    def train_all_classifiers(self):
        """
        Trains and evaluates all classifiers in the list and returns the evaluation metrics.
        
        Returns:
        dict: A dictionary containing the evaluation metrics for each classifier.
        """
        results = {}
        results['SVM (Linear)'] = self.train_svm(kernel='linear')
        results['SVM (RBF)'] = self.train_svm(kernel='rbf')
        results['KNN'] = self.train_knn()
        results['Random Forest'] = self.train_random_forest()
        results['Decision Tree'] = self.train_decision_tree()
        results['Naive Bayes'] = self.train_naive_bayes()
        return results

# Usage Example
if __name__ == "__main__":
    # Example dataset (replace with your actual dataset)
    data = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Feature3': np.random.rand(100),
        'Target': np.random.choice([0, 1], size=100)
    })
    
    # Specify the target column
    target_column = 'Target'

    # Initialize the Classifiers object
    clf = Classifiers(data, target_column)

    # Train and evaluate all classifiers
    results = clf.train_all_classifiers()

    # Print evaluation metrics for each classifier
    for classifier, metrics in results.items():
        print(f"--- {classifier} ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("\n")
