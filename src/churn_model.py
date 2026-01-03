import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_synthetic_data(n_samples=2000):
    """
    Generate synthetic customer churn dataset with realistic features.
    Includes intentional class imbalance to simulate real-world scenario.
    """
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Tenure': np.random.randint(1, 73, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'Usage': np.random.uniform(0, 500, n_samples),
        'Complaints': np.random.randint(0, 8, n_samples),
        'ContractType': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples, p=[0.5, 0.3, 0.2]),
        'PaymentMethod': np.random.choice(['Electronic', 'Mailed Check', 'Bank Transfer', 'Credit Card'], n_samples),
        'CustomerServiceCalls': np.random.randint(0, 10, n_samples),
        'Age': np.random.randint(18, 75, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    churn_probability = (
        (df['Complaints'] * 0.15) +
        (df['CustomerServiceCalls'] * 0.10) +
        ((73 - df['Tenure']) * 0.008) +
        (df['MonthlyCharges'] * 0.005) +
        ((500 - df['Usage']) * 0.001) +
        (df['ContractType'].map({'Month-to-Month': 0.3, 'One Year': 0.1, 'Two Year': 0.05}))
    )
    
    churn_probability = np.clip(churn_probability / churn_probability.max(), 0, 0.85)
    df['Churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
    
    return df

def preprocess_data(df):
    """
    Preprocess data: encode categorical variables and prepare features.
    """
    df_processed = df.copy()
    
    df_processed['ContractType_OneYear'] = (df_processed['ContractType'] == 'One Year').astype(int)
    df_processed['ContractType_TwoYear'] = (df_processed['ContractType'] == 'Two Year').astype(int)
    
    payment_dummies = pd.get_dummies(df_processed['PaymentMethod'], prefix='Payment', drop_first=True)
    df_processed = pd.concat([df_processed, payment_dummies], axis=1)
    
    df_processed = df_processed.drop(['CustomerID', 'ContractType', 'PaymentMethod'], axis=1)
    
    return df_processed

def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train baseline Random Forest without class imbalance handling.
    """
    print("\n" + "="*60)
    print("BASELINE MODEL (No Class Imbalance Handling)")
    print("="*60)
    
    rf_baseline = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_baseline):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_baseline):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_baseline):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_baseline):.4f}")
    
    return rf_baseline, y_pred_baseline

def train_improved_model_smote(X_train, y_train, X_test, y_test):
    """
    Train improved Random Forest with SMOTE for class imbalance.
    """
    print("\n" + "="*60)
    print("IMPROVED MODEL (SMOTE)")
    print("="*60)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"\nOriginal training set distribution: {np.bincount(y_train)}")
    print(f"SMOTE resampled distribution: {np.bincount(y_train_smote)}")
    
    rf_smote = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = rf_smote.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred_smote):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_smote):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_smote):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_smote):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_smote):.4f}")
    
    return rf_smote, y_pred_smote

def train_improved_model_weighted(X_train, y_train, X_test, y_test):
    """
    Train improved Random Forest with class weighting.
    """
    print("\n" + "="*60)
    print("IMPROVED MODEL (Class Weighting)")
    print("="*60)
    
    rf_weighted = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_weighted.fit(X_train, y_train)
    y_pred_weighted = rf_weighted.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_weighted):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_weighted):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_weighted):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_weighted):.4f}")
    
    return rf_weighted, y_pred_weighted

def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_smote, y_pred_weighted):
    """
    Plot confusion matrices for all three models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    cm_smote = confusion_matrix(y_test, y_pred_smote)
    cm_weighted = confusion_matrix(y_test, y_pred_weighted)
    
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('Baseline Model\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
    axes[1].set_title('SMOTE Model\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    sns.heatmap(cm_weighted, annot=True, fmt='d', cmap='Oranges', ax=axes[2], cbar=False)
    axes[2].set_title('Class Weighted Model\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion matrices saved to outputs/confusion_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for the best model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance - Churn Prediction Model', fontsize=16, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved to outputs/feature_importance.png")
    plt.close()

def plot_roc_curves(y_test, rf_baseline, rf_smote, rf_weighted, X_test):
    """
    Plot ROC curves comparing all three models.
    """
    plt.figure(figsize=(10, 8))
    
    y_pred_proba_baseline = rf_baseline.predict_proba(X_test)[:, 1]
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_proba_baseline)
    auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)
    
    y_pred_proba_smote = rf_smote.predict_proba(X_test)[:, 1]
    fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred_proba_smote)
    auc_smote = roc_auc_score(y_test, y_pred_proba_smote)
    
    y_pred_proba_weighted = rf_weighted.predict_proba(X_test)[:, 1]
    fpr_weighted, tpr_weighted, _ = roc_curve(y_test, y_pred_proba_weighted)
    auc_weighted = roc_auc_score(y_test, y_pred_proba_weighted)
    
    plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC = {auc_baseline:.3f})', linewidth=2)
    plt.plot(fpr_smote, tpr_smote, label=f'SMOTE (AUC = {auc_smote:.3f})', linewidth=2)
    plt.plot(fpr_weighted, tpr_weighted, label=f'Class Weighted (AUC = {auc_weighted:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curves saved to outputs/roc_curves.png")
    plt.close()

def main():
    print("="*60)
    print("CUSTOMER CHURN PREDICTION SYSTEM")
    print("="*60)
    
    print("\n[1/7] Generating synthetic customer data...")
    df = generate_synthetic_data(n_samples=2000)
    df.to_csv('data/churn_data.csv', index=False)
    print(f"✓ Dataset created: {len(df)} records")
    print(f"✓ Churn distribution: {df['Churn'].value_counts().to_dict()}")
    
    print("\n[2/7] Preprocessing data...")
    df_processed = preprocess_data(df)
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    feature_names = X.columns.tolist()
    print(f"✓ Features: {len(feature_names)}")
    
    print("\n[3/7] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    print("\n[4/7] Training baseline model...")
    rf_baseline, y_pred_baseline = train_baseline_model(X_train, y_train, X_test, y_test)
    
    print("\n[5/7] Training improved model with SMOTE...")
    rf_smote, y_pred_smote = train_improved_model_smote(X_train, y_train, X_test, y_test)
    
    print("\n[6/7] Training improved model with class weighting...")
    rf_weighted, y_pred_weighted = train_improved_model_weighted(X_train, y_train, X_test, y_test)
    
    print("\n[7/7] Generating visualizations and saving models...")
    
    plot_confusion_matrices(y_test, y_pred_baseline, y_pred_smote, y_pred_weighted)
    
    best_model = rf_smote
    best_name = "SMOTE"
    best_recall = recall_score(y_test, y_pred_smote)
    
    if recall_score(y_test, y_pred_weighted) > best_recall:
        best_model = rf_weighted
        best_name = "Class Weighted"
    
    plot_feature_importance(best_model, feature_names)
    plot_roc_curves(y_test, rf_baseline, rf_smote, rf_weighted, X_test)
    
    joblib.dump(best_model, 'models/rf_churn_model.joblib')
    print(f"\n✓ Best model ({best_name}) saved to models/rf_churn_model.joblib")
    
    print("\n" + "="*60)
    print("SUMMARY - KEY IMPROVEMENTS")
    print("="*60)
    print(f"Baseline Recall: {recall_score(y_test, y_pred_baseline):.4f}")
    print(f"SMOTE Recall: {recall_score(y_test, y_pred_smote):.4f}")
    print(f"Weighted Recall: {recall_score(y_test, y_pred_weighted):.4f}")
    print(f"\nBest Model: {best_name}")
    print("="*60)

if __name__ == "__main__":
    main()