import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class MLProject:
    def __init__(self, task_type='classification'):
        """
        Initialize ML project

        Parameters:
        -----------
        task_type : str, optional (default='classification')
            Type of machine learning task: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        self.pipeline = None

        # Selecting dataset based on the task type
        if self.task_type == 'classification':
            print("Loading Iris dataset for classification task...")
            from sklearn.datasets import load_iris
            dataset = load_iris()
            self.data = pd.DataFrame(data=np.c_[dataset['data'], dataset['target']],
                                    columns=dataset['feature_names'] + ['target'])
            self.target_column = 'target'
            print(f"Dataset loaded with {self.data.shape[0]} samples and {self.data.shape[1]} features.")
        else:
            print("Loading Boston Housing dataset for regression task...")
            from sklearn.datasets import fetch_california_housing
            dataset = fetch_california_housing()
            self.data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
            self.data['target'] = dataset.target
            self.target_column = 'target'
            print(f"Dataset loaded with {self.data.shape[0]} samples and {self.data.shape[1]} features.")

    def exploratory_data_analysis(self):
        """Perform exploratory data analysis on the dataset"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)

        # Display basic information
        print("\nDataset Overview:")
        print(f"Dataset shape: {self.data.shape}")
        print("\nFirst 5 rows:")
        print(self.data.head())

        # Data types and missing values
        print("\nData Types and Missing Values:")
        print(self.data.info())

        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe().T)

        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing Values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found in the dataset.")

        # Distribution of target variable
        plt.figure(figsize=(10, 6))
        if self.task_type == 'classification':
            target_counts = self.data[self.target_column].value_counts()
            sns.countplot(x=self.target_column, data=self.data)
            plt.title('Distribution of Target Classes')
            plt.xlabel('Class')
            plt.ylabel('Count')
            for i, count in enumerate(target_counts):
                plt.text(i, count + 5, f"{count}", ha='center')
        else:
            sns.histplot(self.data[self.target_column], kde=True)
            plt.title('Distribution of Target Variable')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        # Feature distributions
        print("\nFeature Distributions:")
        feature_cols = self.data.columns.drop(self.target_column)

        # Show distributions for a subset of features if there are many
        if len(feature_cols) > 6:
            selected_features = feature_cols[:6]
            print(f"Showing distributions for {len(selected_features)} out of {len(feature_cols)} features.")
        else:
            selected_features = feature_cols

        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(selected_features):
            plt.subplot(2, 3, i+1)
            sns.histplot(self.data[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()
        plt.show()

        # Feature correlations
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

        # Pairplot for classification (limited to 4 features for clarity)
        if self.task_type == 'classification':
            print("\nPairplot of Features by Class:")
            if len(feature_cols) > 4:
                plot_features = feature_cols[:4]
                print(f"Showing pairplot for {len(plot_features)} out of {len(feature_cols)} features.")
            else:
                plot_features = feature_cols

            plt.figure(figsize=(12, 10))
            pairplot_data = self.data.copy()
            if self.task_type == 'classification':
                # Convert numerical target to categorical for better visualization
                pairplot_data[self.target_column] = pairplot_data[self.target_column].astype('category')

            sns.pairplot(pairplot_data, vars=plot_features, hue=self.target_column, height=2.5)
            plt.suptitle('Pairplot of Features by Target Class', y=1.02)
            plt.tight_layout()
            plt.show()

        # Feature importance for understanding relationships
        print("\nFeature Relationship with Target:")
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]

        if self.task_type == 'classification':
            temp_model = RandomForestClassifier(random_state=42, n_estimators=100)
        else:
            temp_model = RandomForestRegressor(random_state=42, n_estimators=100)

        temp_model.fit(X, y)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': temp_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.show()

        print("\nTop 5 Important Features:")
        print(feature_importance.head())

    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data and split into train and test sets

        Parameters:
        -----------
        test_size : float, optional (default=0.2)
            Proportion of data to use for testing
        random_state : int, optional (default=42)
            Random seed for reproducibility
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)

        # Identify numerical and categorical features
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        print(f"\nNumerical features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")

        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Only create categorical transformer if categorical features exist
        if len(categorical_features) > 0:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Create column transformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
        else:
            # If no categorical features, only use numeric transformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ]
            )

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.task_type == 'classification' else None
        )

        print(f"\nTrain set size: {self.X_train.shape[0]} samples")
        print(f"Test set size: {self.X_test.shape[0]} samples")

        return self

    def train_model(self, optimize=True):
        """
        Train the Random Forest model

        Parameters:
        -----------
        optimize : bool, optional (default=True)
            Whether to perform hyperparameter optimization
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)

        if self.task_type == 'classification':
            rf = RandomForestClassifier(random_state=42)

            if optimize:
                print("\nPerforming hyperparameter optimization...")
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }

                # Create pipeline with preprocessing and model
                self.pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('classifier', GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1))
                ])

                # Train the pipeline
                self.pipeline.fit(self.X_train, self.y_train)

                # Get the best model
                best_params = self.pipeline.named_steps['classifier'].best_params_
                print(f"\nBest hyperparameters: {best_params}")

                # Update the model with the best parameters
                self.model = RandomForestClassifier(random_state=42, **best_params)
            else:
                print("\nTraining with default parameters...")
                self.pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('classifier', rf)
                ])
                self.pipeline.fit(self.X_train, self.y_train)
                self.model = rf

        else:  # Regression
            rf = RandomForestRegressor(random_state=42)

            if optimize:
                print("\nPerforming hyperparameter optimization...")
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }

                # Create pipeline with preprocessing and model
                self.pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('regressor', GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1))
                ])

                # Train the pipeline
                self.pipeline.fit(self.X_train, self.y_train)

                # Get the best model
                best_params = self.pipeline.named_steps['regressor'].best_params_
                print(f"\nBest hyperparameters: {best_params}")

                # Update the model with the best parameters
                self.model = RandomForestRegressor(random_state=42, **best_params)
            else:
                print("\nTraining with default parameters...")
                self.pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('regressor', rf)
                ])
                self.pipeline.fit(self.X_train, self.y_train)
                self.model = rf

        # Either way, train the best model on the full training set
        X_train_processed = self.pipeline.named_steps['preprocessor'].transform(self.X_train)
        self.model.fit(X_train_processed, self.y_train)

        print("\nModel training completed.")

        # Cross-validation
        print("\nPerforming cross-validation...")
        cv_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])

        if self.task_type == 'classification':
            cv_scores = cross_val_score(cv_pipeline, self.X_train, self.y_train, cv=5, scoring='accuracy')
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        else:
            cv_scores = cross_val_score(cv_pipeline, self.X_train, self.y_train, cv=5, scoring='r2')
            print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        return self

    def evaluate_model(self):
        """Evaluate the trained model on the test set"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)

        # Transform the test data
        X_test_processed = self.pipeline.named_steps['preprocessor'].transform(self.X_test)

        # Make predictions
        y_pred = self.model.predict(X_test_processed)

        if self.task_type == 'classification':
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"\nTest Accuracy: {accuracy:.4f}")

            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))

            # Confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.show()

            # Feature importance
            feature_names = self.X_train.columns
            feature_importances = self.model.feature_importances_

            # Get feature names after one-hot encoding (if applicable)
            if hasattr(self.pipeline.named_steps['preprocessor'], 'transformers_'):
                cat_features_exist = any(name == 'cat' for name, _, _ in self.pipeline.named_steps['preprocessor'].transformers_)

                if cat_features_exist:
                    # Extract transformed feature names
                    ohe = self.pipeline.named_steps['preprocessor'].named_transformers_.get('cat', None)
                    if ohe is not None and hasattr(ohe, 'named_steps') and 'onehot' in ohe.named_steps:
                        cat_cols = self.pipeline.named_steps['preprocessor'].transformers_[1][2]
                        cat_features = ohe.named_steps['onehot'].get_feature_names_out(cat_cols)
                        num_cols = self.pipeline.named_steps['preprocessor'].transformers_[0][2]
                        feature_names = np.append(num_cols, cat_features)

            # Sort features by importance
            indices = np.argsort(feature_importances)[::-1]

            # Plot feature importance
            plt.figure(figsize=(12, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(indices)), feature_importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()

            # ROC Curve and AUC
            if hasattr(self.model, 'predict_proba'):
                from sklearn.metrics import roc_curve, auc
                y_prob = self.model.predict_proba(X_test_processed)

                # For multi-class, plot ROC for each class
                if y_prob.shape[1] > 2:
                    from sklearn.preprocessing import label_binarize
                    from sklearn.metrics import roc_auc_score

                    # Binarize the output for multi-class
                    classes = np.unique(self.y_test)
                    y_test_bin = label_binarize(self.y_test, classes=classes)

                    # Compute ROC curve and ROC area for each class
                    plt.figure(figsize=(10, 8))

                    for i, class_label in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Multi-class ROC Curve')
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    plt.show()

                    # Calculate micro-average ROC curve and ROC area
                    roc_auc_micro = roc_auc_score(y_test_bin, y_prob, average='micro')
                    print(f"\nMicro-Average AUC: {roc_auc_micro:.4f}")

                # For binary classification
                elif y_prob.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(self.y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)

                    plt.figure(figsize=(10, 8))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    plt.show()

                    print(f"\nAUC: {roc_auc:.4f}")

        else:  # Regression
            # Calculate R-squared
            r2 = r2_score(self.y_test, y_pred)
            print(f"\nTest R²: {r2:.4f}")

            # Calculate Mean Squared Error
            mse = mean_squared_error(self.y_test, y_pred)
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

            # Plot actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            plt.tight_layout()
            plt.show()

            # Plot residuals
            residuals = self.y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.tight_layout()
            plt.show()

            # Residual distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True)
            plt.xlabel('Residual')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            plt.tight_layout()
            plt.show()

            # Feature importance
            feature_importances = self.model.feature_importances_
            feature_names = self.X_train.columns

            # Sort features by importance
            indices = np.argsort(feature_importances)[::-1]

            # Plot feature importance
            plt.figure(figsize=(12, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(indices)), feature_importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()

        return self

    def save_model(self, filename="random_forest_model.pkl"):
        """
        Save the trained model to a file

        Parameters:
        -----------
        filename : str, optional (default="random_forest_model.pkl")
            Filename to save the model
        """
        import pickle

        print("\n" + "="*50)
        print("MODEL SAVING")
        print("="*50)

        with open(filename, 'wb') as f:
            pickle.dump(self.pipeline, f)

        print(f"\nModel saved to {filename}")

        return self

# Example of using the class
if __name__ == "__main__":
    # For classification
    print("\nRunning Classification Task\n")
    clf_project = MLProject(task_type='classification')
    clf_project.exploratory_data_analysis()
    clf_project.preprocess_data()
    clf_project.train_model(optimize=True)
    clf_project.evaluate_model()
    clf_project.save_model("random_forest_classifier.pkl")

    # For regression
    print("\nRunning Regression Task\n")
    reg_project = MLProject(task_type='regression')
    reg_project.exploratory_data_analysis()
    reg_project.preprocess_data()
    reg_project.train_model(optimize=True)
    reg_project.evaluate_model()
    reg_project.save_model("random_forest_regressor.pkl")
