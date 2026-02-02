import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# 6 Different Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Configuration
DATA_FILE = "my_system_data.csv"
MODEL_FILE = "anomaly_detector.pkl"
SCALER_FILE = "scaler.pkl"
REPORT_FILE = "evaluation_report.txt"

def train_and_compare():
    print("\n" + "="*60)
    print("TRAINING & MODEL SELECTION")
    print("="*60)
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} samples from {DATA_FILE}")
    except FileNotFoundError:
        print(f"[!] Error: {DATA_FILE} not found. Run collect_data.py first!")
        return

    # 2. Prepare Features
    X = df.drop(columns=['label'])
    y = df['label']
    
    # 3. Split Data (80% Train, 20% Test)
    # Stratify ensures we have attacks in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Define Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42)
    }
    
    results = []
    
    print(f"\nTraining {len(models)} models...\n")
    print(f"{'MODEL':<20} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 70)
    
    best_model_name = ""
    best_model_obj = None
    best_f1 = -1
    best_report = ""
    
    report_content = "DCML PROJECT - MODEL EVALUATION REPORT\n"
    report_content += "="*50 + "\n\n"
    
    # 6. Train & Evaluate Loop
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Detailed metrics
        report_content += f"Model: {name}\n"
        report_content += f"Accuracy:  {acc:.4f}\n"
        report_content += f"Precision: {prec:.4f}\n"
        report_content += f"Recall:    {rec:.4f}\n"
        report_content += f"F1-Score:  {f1:.4f}\n"
        report_content += "-"*20 + "\n"
        
        print(f"{name:<20} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            # Save the full classification report for the winner
            best_report = classification_report(y_test, y_pred)
            
    print("-" * 70)
    print(f"\nWINNER: {best_model_name} (F1: {best_f1:.4f})")
    
    # 7. Add Winner details to report
    report_content += "\n" + "="*50 + "\n"
    report_content += f"WINNER SELECTION: {best_model_name}\n"
    report_content += "="*50 + "\n"
    report_content += "Detailed Classification Report:\n\n"
    report_content += best_report
    
    # Save Report
    with open(REPORT_FILE, "w") as f:
        f.write(report_content)
    print(f"Full evaluation report saved to: {REPORT_FILE}")
    
    # 8. Save the Winner
    print(f"Saving {best_model_name} to {MODEL_FILE}...")
    joblib.dump(best_model_obj, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    print("\nTraining Complete. Ready for Detection!")

if __name__ == "__main__":
    train_and_compare()
