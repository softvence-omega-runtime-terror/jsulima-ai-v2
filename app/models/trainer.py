"""
UFC Model Trainer Module

This module handles training and evaluation of multiple ML models
for UFC fight prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from typing import Dict, Tuple, Any
import time


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, dict, float]:
    """
    Train Logistic Regression with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
    
    Returns:
        Tuple of (best_model, best_params, training_time)
    """
    param_grid = {
        'C': [0.01, 0.1, 1.2, 10],
        'penalty': ['l1','l2'],
        'solver': ['lbfgs'],
        'max_iter': [500, 1000, 3000]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, dict, float]:
    """
    Train Random Forest with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
    
    Returns:
        Tuple of (best_model, best_params, training_time)
    """
    param_grid = {
        'n_estimators': [200, 450, 600, 1000],
        'max_depth': [10, 14, 30, None],
        'min_samples_split': [2, 4, 10],
        'max_features': ['sqrt', 'log2'],
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, dict, float]:
    """
    Train Gradient Boosting with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
    
    Returns:
        Tuple of (best_model, best_params, training_time)
    """
    param_grid = {
        'n_estimators': [100, 150, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2']
    }
    
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, dict, float]:
    """
    Train Support Vector Machine with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
    
    Returns:
        Tuple of (best_model, best_params, training_time)
    """
    param_grid = {
        'C': [0.1, 2, 5, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(
        SVC(random_state=42, probability=True),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time


def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> Tuple[Any, dict, float]:
    """
    Train K-Nearest Neighbors with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
    
    Returns:
        Tuple of (best_model, best_params, training_time)
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5
) -> Dict[str, dict]:
    """
    Train all models and return results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        cv: Number of cross-validation folds
    
    Returns:
        Dictionary with results for all models
    """
    results = {}
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model, lr_params, lr_time = train_logistic_regression(X_train, y_train, cv)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results['Logistic Regression'] = {
        'model': lr_model,
        'params': lr_params,
        'time': lr_time,
        'metrics': lr_metrics
    }
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model, rf_params, rf_time = train_random_forest(X_train, y_train, cv)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results['Random Forest'] = {
        'model': rf_model,
        'params': rf_params,
        'time': rf_time,
        'metrics': rf_metrics
    }
    
    # Train Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model, gb_params, gb_time = train_gradient_boosting(X_train, y_train, cv)
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    results['Gradient Boosting'] = {
        'model': gb_model,
        'params': gb_params,
        'time': gb_time,
        'metrics': gb_metrics
    }
    
    # Train SVM
    print("Training SVM...")
    svm_model, svm_params, svm_time = train_svm(X_train, y_train, cv)
    svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM")
    results['SVM'] = {
        'model': svm_model,
        'params': svm_params,
        'time': svm_time,
        'metrics': svm_metrics
    }
    
    # Train KNN
    print("Training KNN...")
    knn_model, knn_params, knn_time = train_knn(X_train, y_train, cv)
    knn_metrics = evaluate_model(knn_model, X_test, y_test, "KNN")
    results['KNN'] = {
        'model': knn_model,
        'params': knn_params,
        'time': knn_time,
        'metrics': knn_metrics
    }
    
    return results


def select_best_model(results: Dict[str, dict]) -> Tuple[str, Any, dict]:
    """
    Select the best model based on accuracy.
    
    Args:
        results: Dictionary with results from all models
    
    Returns:
        Tuple of (model_name, model, metrics)
    """
    best_name = None
    best_accuracy = 0
    best_model = None
    best_metrics = None
    
    for model_name, model_data in results.items():
        accuracy = model_data['metrics']['accuracy']
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = model_name
            best_model = model_data['model']
            best_metrics = model_data['metrics']
    
    return best_name, best_model, best_metrics
