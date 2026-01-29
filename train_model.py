"""
DCML Project - Model Training Script
Socket Drain Attack Anomaly Detector

This script trains and compares multiple ML algorithms for detecting Socket Drain attacks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = "my_system_data.csv"
MODEL_FILE = "anomaly_detector.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES = ["cpu", "ram", "sockets", "net_speed", "threads"]
TARGET = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_prepare_data():
    """Load CSV and prepare features/target"""
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples")
    print(f"Features: {FEATURES}")
    print(f"Class distribution:")
    print(f"  - Normal (0): {len(df[df[TARGET] == 0])}")
    print(f"  - Anomaly (1): {len(df[df[TARGET] == 1])}")
    
    X = df[FEATURES]
    y = df[TARGET]
    
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    print("\n" + "=" * 60)
    print("STEP 2: Training & Evaluating Models")
    print("=" * 60)
    
    # Define models to compare
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel='rbf', random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "CV Mean": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "model_obj": model
        })
        
        print(f"  Accuracy: {accuracy:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results

def select_best_model(results):
    """Select the best model based on F1-Score"""
    print("\n" + "=" * 60)
    print("STEP 3: Model Comparison & Selection")
    print("=" * 60)
    
    # Sort by F1-Score
    results_sorted = sorted(results, key=lambda x: x["F1-Score"], reverse=True)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1-Score"
    ))
    print("-" * 65)
    
    for r in results_sorted:
        print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r["Model"], r["Accuracy"], r["Precision"], r["Recall"], r["F1-Score"]
        ))
    
    best = results_sorted[0]
    print(f"\n*** BEST MODEL: {best['Model']} ***")
    print(f"    F1-Score: {best['F1-Score']:.4f}")
    print(f"    Accuracy: {best['Accuracy']:.4f}")
    
    return best

def save_model(model, scaler):
    """Save the best model and scaler to files"""
    print("\n" + "=" * 60)
    print("STEP 4: Saving Model")
    print("=" * 60)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Scaler saved to: {SCALER_FILE}")

def main():
    print("\n" + "#" * 60)
    print("#  DCML Socket Drain Anomaly Detector - Model Training")
    print("#" * 60)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Select best model
    best = select_best_model(results)
    
    # Save best model
    save_model(best["model_obj"], scaler)
    
    print("\n" + "#" * 60)
    print("#  Training Complete!")
    print("#" * 60)
    print(f"\nYour anomaly detector is ready: {MODEL_FILE}")
    print("Use this model in your runtime detector to detect Socket Drain attacks.\n")

if __name__ == "__main__":
    main()
