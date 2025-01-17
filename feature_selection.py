from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import pandas as pd
from data_cleaning import DatasetCleaner
from sklearn.base import clone
import numpy as np
from sklearn.feature_selection import mutual_info_classif

class FeatureSelection:
    def __init__(self, df):
        self.df = df

    def split_target(self):
        # Split feature and target vectors  
        X = self.df.drop("target", axis=1)
        Y = self.df["target"]
        return X, Y

    def split_data (self):
        # Split train and test sets
        """
        Split the data into training and test sets.

        Parameters:
        X (pd.DataFrame): Features data.
        Y (pd.Series): Target variable.

        Returns:
        X_train, X_test, Y_train, Y_test (tuple of pd.DataFrame, pd.Series):
            The training and test data split into their respective design and target matrices.
        """
        X, Y = self.split_target()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 42)
        return X_train, X_test, Y_train, Y_test
        

    def corr_with_label (self, threshold = 0.2):
        
        """
        Identify features in the DataFrame that have a correlation with the target variable
        exceeding a given threshold.
        
        This function calculates the correlation matrix of the DataFrame, extracts the absolute 
        correlation values with respect to the 'target' column, and selects the features whose 
        correlation values exceed the specified threshold. The 'target' variable itself is 
        excluded from the results.

        Parameters:
        threshold (float): The correlation threshold above which features are selected. 
                        Default is 0.2.

        Returns:
        list: A list of feature names that are highly correlated with the target variable.
        """

        cor = self.df.corr()    

        # Get the absolute value of the correlation
        cor_target = abs(cor["target"])

        # Select highly correlated features (thresold = 0.2)
        relevant_features = cor_target[cor_target > threshold]

        # Collect the names of the features
        selected_features = [index for index, value in relevant_features.items()]

        # Drop the target variable from the results
        selected_features.remove('target')

        return selected_features
    
    
    def anova_selection(self, k = 10):
        """
        Select top k features based on ANOVA F-value.
        
        This function uses the SelectKBest from scikit-learn to select the top k features
        according to the ANOVA F-value. The function first standardizes the features using
        the StandardScaler, then applies the SelectKBest to select the top k features.
        
        Parameters:
        k (int): The number of features to select. Default is 10.
        
        Returns:
        list: A list of feature names that are selected by the ANOVA F-value.
        """
        X, Y = self.split_target()
        X_scaled, _ = DatasetCleaner.standardization (X)
        # print(X_scaled.shape)
        selector = SelectKBest(f_classif, k= min(2, X.shape[1]))
        # Fit to scaled data, then transform it
        _ = selector.fit_transform(X_scaled, Y)
    
        feature_names = X.columns[selector.get_support()] 
        return feature_names
    

    def run_rfe(self):
    
        """
        Select top k features based on Recursive Feature Elimination (RFE) using a
        RandomForestClassifier.

        This function iterates over possible number of features (from max to 2) and
        selects the best k features using RFE with the provided model. The function
        then returns the feature names of the selected features.

        Parameters:
        None

        Returns:
        list: A list of feature names that are selected by RFE.
        """
        
        X_train, X_test, Y_train, Y_test = self.split_data()
        X_train, X_test = DatasetCleaner.standardization(X_train, X_test)

        best_score = -1  # Start with a score lower than the possible minimum
        best_rfe = None  # To store the RFE model with the best k

        model = RandomForestClassifier(criterion='entropy', random_state=47)

        # Iterate over possible number of features (from max to 2)
        for k in range(X_train.shape[1], 1, -1):
            # Initialize RFE with the model and desired number of features
            rfe = RFE(estimator=model, n_features_to_select=k)
            
            # Fit RFE on the training data
            rfe.fit(X_train, Y_train)
            
            # Predict on the test data using the RFE model
            y_pred = rfe.predict(X_test)
            
            # Calculate the F1 score
            score = f1_score(Y_test, y_pred)
            
            # Update best_score and best_k if current score is better
            if score > best_score:
                best_score = score
                best_rfe = rfe
            
            # print(f"k={k}, F1 Score={score:.4f}") 

        feature_names = self.df.drop("target",axis=1).columns[rfe.get_support()]
        return feature_names


    def feature_importances_from_tree_based_model_(self):
        
        """
        Fit a Random Forest Classifier to the data and return the model.

        Parameters:
        None

        Returns:
        model (RandomForestClassifier): The trained model.
        """
        X, Y = self.split_target()
        # X_train, X_test, Y_train, Y_test = self.split_data (X, Y)

        model = RandomForestClassifier()
        model = model.fit(X, Y)
        
        return model


    def select_features_from_model(self, model, threshold=0.013):
        
        model = SelectFromModel(model, prefit=True, threshold= threshold)
        feature_idx = model.get_support()
        feature_names = self.df.drop("target",axis=1).columns[feature_idx]

        return feature_names

    def run_l1_regularization(self):
        
        X, Y = self.split_target()
        X_train, X_test, Y_train, Y_test = self.split_data()
        X_train, _ = DatasetCleaner.standardization(X_train, X_test)
        
        # Select L1 regulated features from LinearSVC output 
        selection = SelectFromModel(LinearSVC(C=1, penalty='l1', dual=False))
        selection.fit(X_train, Y_train)

        feature_names = self.df.drop("target",axis= 1 ).columns[(selection.get_support())]
        
        return feature_names
    

    def forward_feature_selection(self, scoring=accuracy_score):
        """
        Forward Feature Selection Algorithm.

        Parameters:
        - model: A scikit-learn model
        - X: Feature matrix (numpy array or pandas DataFrame)
        - y: Target vector
        - scoring: Scoring function (default is accuracy_score)

        Returns:
        - selected_features: List of selected feature indices
        """
        model = RandomForestClassifier(criterion='entropy', random_state=47)
        X, Y = self.split_target()
        X_train, X_test, Y_train, Y_test = self.split_data()
        n_features = X.shape[1]
        selected_features = []
        remaining_features = list(X_train.columns)
        # print(remaining_features)
        best_score = -1

        while remaining_features:
            scores = []
            for feature in remaining_features:
                # print(feature)
                candidate_features = selected_features + [feature]
                # print(candidate_features)
                model_clone = clone(model)
                # print(X_train[candidate_features])
                model_clone.fit(X_train[candidate_features], Y_train)
                y_pred = model_clone.predict(X_test[candidate_features])
                score = scoring(Y_test, y_pred)
                scores.append((score, candidate_features))

            scores.sort(reverse=True, key=lambda x: x[0])
            best_candidate_score, best_candidate = scores[0]

            if best_candidate_score > best_score:
                best_score = best_candidate_score
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
            else:
                break
        
        return selected_features

    def backward_feature_elimination(model, X, y, scoring=accuracy_score):
        """
        Backward Feature Elimination Algorithm.

        Parameters:
        - model: A scikit-learn model
        - X: Feature matrix (numpy array or pandas DataFrame)
        - y: Target vector
        - scoring: Scoring function (default is accuracy_score)

        Returns:
        - selected_features: List of selected feature indices
        """
        n_features = X.shape[1]
        selected_features = list(range(n_features))
        best_score = -np.inf

        while len(selected_features) > 1:
            scores = []
            for feature in selected_features:
                candidate_features = [f for f in selected_features if f != feature]
                model_clone = clone(model)
                model_clone.fit(X[:, candidate_features], y)
                y_pred = model_clone.predict(X[:, candidate_features])
                score = scoring(y, y_pred)
                scores.append((score, feature))

            scores.sort(reverse=True, key=lambda x: x[0])
            best_candidate_score, worst_feature = scores[0]

            if best_candidate_score > best_score:
                best_score = best_candidate_score
                selected_features.remove(worst_feature)
            else:
                break

        return selected_features
        

    def evaluate_model_on_features(self, names_of_features):
        
        #Split the data into training and test sets.
        X, Y = self.split_target()
        

        '''Train model and display evaluation metrics.'''
        def fit_model(X, Y):
            '''Use a RandomForestClassifier for this problem.'''
            
            # define the model to use
            model = RandomForestClassifier(criterion='entropy', random_state=47)
            
            # Train the model
            model.fit(X, Y)
            
            return model

        def calculate_metrics(model, X_test_scaled, Y_test):
            '''Get model evaluation metrics on the test set.'''
            
            # Get model predictions
            y_predict_r = model.predict(X_test_scaled)
            
            # Calculate evaluation metrics for assesing performance of the model.
            acc = accuracy_score(Y_test, y_predict_r)
            roc = roc_auc_score(Y_test, y_predict_r)
            prec = precision_score(Y_test, y_predict_r)
            rec = recall_score(Y_test, y_predict_r)
            f1 = f1_score(Y_test, y_predict_r)
            
            return acc, roc, prec, rec, f1
        
        def train_and_get_metrics(X, Y):
            
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state = 123)
    
            X_train, X_test = DatasetCleaner.standardization (X_train, X_test)

            # Call the fit model function to train the model on the normalized features and the diagnosis values
            model = fit_model(X_train, Y_train)

            # Make predictions on test dataset and calculate metrics.
            acc, roc, prec, rec, f1 = calculate_metrics(model, X_test, Y_test)

            return acc, roc, prec, rec, f1

        # Train the model, predict values and get metrics
        acc, roc, prec, rec, f1 = train_and_get_metrics(X[names_of_features], Y)

        # Construct a dataframe to display metrics.
        display_df = pd.DataFrame([[acc, roc, prec, rec, f1, X[names_of_features].shape[1]]], columns=["Accuracy", "ROC", "Precision", "Recall", "F1 Score", 'Feature Count'])
        
        return display_df
    
    def feature_selection_with_mutual_information(self, top_n=10):
        """
        Perform feature selection using mutual information and evaluate performance with cross-validation.

        Parameters:
            features (pd.DataFrame or np.ndarray): Input features.
            target (pd.Series or np.ndarray): Target variable.
            problem_type (str): Either 'classification' or 'regression'.
            top_n (int): Number of top features to select based on mutual information.
            cv_folds (int): Number of folds for cross-validation.

        Returns:
            dict: Contains selected features, mutual information scores, and cross-validation accuracy.
        """
        X, Y = self.split_target()
        mi_scores = mutual_info_classif(X, Y, random_state=42)
        
        # Rank features by MI scores
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'Feature_{i}' for i in range(X.shape[1])]
        feature_ranking = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
        feature_ranking = feature_ranking.sort_values(by='MI_Score', ascending=False)

        # Select top-N features
        selected_features = feature_ranking.head(top_n)['Feature']

        return selected_features.tolist()
    

    def feature_selection_table(self):
        
        # All features
        results = self.evaluate_model_on_features(self.df.drop("target", axis=1).columns)
        results.index = ['All features']


        # # corr with label
        corr_slected_features = self.corr_with_label (threshold=0.2)
        strong_features_eval_df = self.evaluate_model_on_features(corr_slected_features)
        strong_features_eval_df.index = ['cor with target']
        results = pd.concat([results, strong_features_eval_df])

        mi_selected_features = self.feature_selection_with_mutual_information(4)
        mi_features_eval_df = self.evaluate_model_on_features(mi_selected_features)
        mi_features_eval_df.index = ['mutual information']
        results = pd.concat([results, mi_features_eval_df])

        # # anova f-test
        anova_selected_features = self.anova_selection(k = 5)
        univariate_eval_df = self.evaluate_model_on_features(anova_selected_features)
        univariate_eval_df.index = ['anova F-test']
        results = pd.concat([results, univariate_eval_df])

        # # Recursive Feature Elimination
        rfe_feature_names = self.run_rfe()
        rfe_eval_df = self.evaluate_model_on_features(rfe_feature_names)
        rfe_eval_df.index = ['RFE']
        results = pd.concat([results, rfe_eval_df])

        # # Feature Importance
        model = self.feature_importances_from_tree_based_model_()
        feature_imp_feature_names = self.select_features_from_model(model)
        fimp_eval_df = self.evaluate_model_on_features(feature_imp_feature_names)
        fimp_eval_df.index = ['Feature Importance']
        results = pd.concat([results, fimp_eval_df])

        # # L1 Regularization
        l1reg_feature_names = self.run_l1_regularization()
        l1reg_eval_df = self.evaluate_model_on_features(l1reg_feature_names)
        l1reg_eval_df.index = ['L1 Reg']
        results = pd.concat([results, l1reg_eval_df])

        print(results)
