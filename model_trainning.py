# Name of the file: Model_trainning.ipynb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import time

base_url = "https://raw.githubusercontent.com/pranta-iitp/Duplicate-Question-Pairs-Detection/main/data/"
file_names = [f"final_features_part_{i}.csv" for i in range(1, 51)]
urls = [base_url + name for name in file_names]
dfs = [pd.read_csv(url) for url in urls]
df = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(df)} samples")
print(df.head())




def nlp_scorer(model_name, model, preprocessor, X, y):
    """Enhanced model evaluation function for binary classification"""
    output = []
    output.append(model_name)

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # K-fold cross-validation with multiple metrics
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold for NLP tasks

    # Cross-validation scores
    cv_accuracy = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
    cv_f1 = cross_val_score(pipeline, X, y, cv=kfold, scoring='f1')
    cv_roc_auc = cross_val_score(pipeline, X, y, cv=kfold, scoring='roc_auc')

    # CV metrics mean and std
    output.extend([cv_accuracy.mean(), cv_accuracy.std()])
    output.extend([cv_f1.mean(), cv_f1.std()])
    output.extend([cv_roc_auc.mean(), cv_roc_auc.std()])

    # Train-test split for additional metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Measure training time
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability for positive class

    # Test set metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)

    output.extend([test_accuracy, test_precision, test_recall, test_f1, test_roc_auc])
    output.append(training_time)

    return output

model_dict = {
    'random_forest': RandomForestClassifier(random_state=42, n_estimators=50),
    'extra_trees': ExtraTreesClassifier(random_state=42, n_estimators=50),
    'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=50),
    'adaboost': AdaBoostClassifier(random_state=42, n_estimators=50),
    'xgboost': XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50)
}
import gc
def run_classification_experiments(df, target_column, scale_features=True, sample_size=5000):
    # Add this after the function definition line
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using sample of {sample_size} rows due to memory constraints")
    

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if scale_features:
        preprocessor = StandardScaler()
    else:
        preprocessor = None  # No preprocessing needed

    # Train models sequentially to save memory

    results_list = []

    for model_name, model in model_dict.items():
        print(f"Training {model_name}...")
        result = nlp_scorer(model_name, model, preprocessor, X, y)
        
        # Print immediately and store minimal info
        print(f"{model_name} - F1: {result[4]:.3f}, Accuracy: {result[1]:.3f}")
        results_list.append(result)
        
        # Clear memory after each model
        del result, model
        gc.collect()


    results_df = pd.DataFrame(results_list, columns=columns)
    return results_df.sort_values(['CV_F1_Mean'], ascending=False)


model_output = run_classification_experiments(df, 'is_duplicate')

# Create results DataFrame
columns = [
    'Model',
    'CV_Accuracy_Mean', 'CV_Accuracy_Std',
    'CV_F1_Mean', 'CV_F1_Std',
    'CV_ROC_AUC_Mean', 'CV_ROC_AUC_Std',
    'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC',
    'Training_Time'
]

results_df = pd.DataFrame(model_output, columns=columns)

# Sort by F1 score (good for binary classification)
print(results_df.sort_values(['CV_F1_Mean'], ascending=False))

#output
"""Using sample of 5000 rows due to memory constraints
Training random_forest...
random_forest - F1: 0.017, Accuracy: 0.751
Training extra_trees...
extra_trees - F1: 0.013, Accuracy: 0.758
Training gradient_boosting...
gradient_boosting - F1: 0.021, Accuracy: 0.745
Training adaboost...
adaboost - F1: 0.037, Accuracy: 0.712
Training xgboost...
xgboost - F1: 0.020, Accuracy: 0.742"""
import optuna
from sklearn.model_selection import cross_val_score

def optimize_random_forest(X, y, n_trials=100):
    """
    Hyperparameter tuning for Random Forest using Optuna
    """
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100,200,300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        
        # Create model with suggested parameters
        model = RandomForestClassifier(**params)
        
        # Use cross-validation to evaluate
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='f1', n_jobs=-1)
        
        return cv_scores.mean()
    
    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("Best parameters:", study.best_params)
    print("Best F1 score:", study.best_value)
    
    return study.best_params
# Before running all models, optimize Random Forest
print("Optimizing Random Forest hyperparameters...")
X = df.drop(columns=['is_duplicate'])
y = df['is_duplicate']

df_sample = df.sample(n=20000, random_state=42)
X_sample = df_sample.drop(columns=['is_duplicate'])
y_sample = df_sample['is_duplicate']
# Get best parameters
best_rf_params = optimize_random_forest(X_sample, y_sample)

# Update your model_dict with optimized Random Forest
model_dict['random_forest'] = RandomForestClassifier(**best_rf_params)