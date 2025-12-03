"""
UFC Model Training Script

This script trains the UFC winner prediction model using the modular pipeline.
Run this script to train and save the model artifacts.

Usage:
    python scripts/train_model.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from app.services.loader import load_ufc_data, validate_data_schema, get_data_info
from app.services.preprocessor import preprocess_pipeline
from app.models.trainer import train_all_models, select_best_model
from app.utils.model_utils import (
    save_model, save_scaler, save_feature_names, save_model_metadata
)


def main():
    """Main training pipeline."""
    
    print("=" * 80)
    print("UFC WINNER PREDICTION MODEL TRAINING")
    print("=" * 80)
    
    # Step 1: Load Data
    print("\n" + "=" * 80)
    print("STEP 1: Loading UFC Data")
    print("=" * 80)
    
    df = load_ufc_data()
    print(f"[OK] Loaded data: {df.shape}")
    
    # Validate schema
    validate_data_schema(df)
    print("[OK] Data schema validated")
    
    # Show data info
    data_info = get_data_info(df)
    print(f"\nData Info:")
    print(f"  Total fights: {data_info['total_fights']}")
    print(f"  Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
    print(f"  Total columns: {data_info['total_columns']}")
    print(f"  Missing values: {data_info['missing_values']}")
    
    # Step 2: Preprocess Data
    print("\n" + "=" * 80)
    print("STEP 2: Preprocessing Data")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)
    
    print(f"[OK] Data preprocessed")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {len(X_train.columns)}")
    
    # Display feature names
    print(f"\nFeatures ({len(X_train.columns)}):")
    for i, col in enumerate(X_train.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Display class distribution
    print(f"\nClass Distribution:")
    print(f"  Training - Local Winner: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"  Training - Away Winner: {(len(y_train) - y_train.sum())} ({(len(y_train) - y_train.sum())/len(y_train)*100:.1f}%)")
    print(f"  Test - Local Winner: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    print(f"  Test - Away Winner: {(len(y_test) - y_test.sum())} ({(len(y_test) - y_test.sum())/len(y_test)*100:.1f}%)")
    
    # Step 3: Train Models
    print("\n" + "=" * 80)
    print("STEP 3: Training Models")
    print("=" * 80)
    
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': data['metrics']['accuracy'],
            'Precision': data['metrics']['precision'],
            'Recall': data['metrics']['recall'],
            'F1-Score': data['metrics']['f1_score'],
            'ROC-AUC': data['metrics']['roc_auc'],
            'Training Time (s)': data['time']
        }
        for name, data in results.items()
    ])
    
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=False)
    print("\n" + results_df.to_string(index=False))
    
    # Step 4: Select Best Model
    print("\n" + "=" * 80)
    print("STEP 4: Selecting Best Model")
    print("=" * 80)
    
    best_name, best_model, best_metrics = select_best_model(results)
    
    print(f"\n[BEST MODEL]: {best_name}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    # Step 5: Save Model Artifacts
    print("\n" + "=" * 80)
    print("STEP 5: Saving Model Artifacts")
    print("=" * 80)
    
    # Save model
    model_path = save_model(best_model)
    print(f"[OK] Model saved: {model_path}")
    
    # Save scaler
    scaler_path = save_scaler(scaler)
    print(f"[OK] Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = save_feature_names(X_train.columns.tolist())
    print(f"[OK] Feature names saved: {features_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_name,
        'accuracy': best_metrics['accuracy'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1_score': best_metrics['f1_score'],
        'roc_auc': best_metrics['roc_auc'],
        'num_features': len(X_train.columns),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'model_params': results[best_name]['params']
    }
    metadata_path = save_model_metadata(metadata)
    print(f"[OK] Metadata saved: {metadata_path}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("[SUCCESS] TRAINING COMPLETE")
    print("=" * 80)
    
    print(f"\nModel Artifacts:")
    print(f"  • Model: {model_path}")
    print(f"  • Scaler: {scaler_path}")
    print(f"  • Features: {features_path}")
    print(f"  • Metadata: {metadata_path}")
    
    print(f"\nModel Performance:")
    print(f"  • Accuracy: {best_metrics['accuracy']:.2%}")
    print(f"  • ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    print(f"\nThe model is ready for deployment! [READY]")
    print("\nTo use the model:")
    print("  1. Start the FastAPI server")
    print("  2. Make predictions via POST /predict")
    print("  3. Get model info via GET /model/info")


if __name__ == "__main__":
    main()
