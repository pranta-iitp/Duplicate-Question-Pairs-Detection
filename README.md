# Quora Duplicate Question Detection

A comprehensive machine learning project for detecting duplicate question pairs using advanced NLP techniques and ensemble methods.

## 📋 Project Overview

This project implements a binary classification system to identify whether two questions from Quora are duplicates. The dataset is sourced from [Kaggle - Quora Duplicate Questions Detection](https://www.kaggle.com/datasets/thedevastator/quora-duplicate-questions-detection/data). The pipeline consists of two main components:
1. **Feature Engineering & Preprocessing** 
2. **Model Training & Hyperparameter Optimization**

## 🚀 Project Structure

```
├── data/
│   ├── train_part_1.csv                      # Raw training data (part 1)
│   ├── train_part_2.csv                      # Raw training data (part 2)
│   ├── train_part_3.csv                      # Raw training data (part 3)
│   ├── train_part_4.csv                      # Raw training data (part 4)
│   ├── train_part_5.csv                      # Raw training data (part 5)
│   ├── final_features_part_1.csv             # Processed features (part 1)
│   ├── final_features_part_2.csv             # Processed features (part 2)
│   ├── ...                                   # ... (parts 3-49)
│   ├── final_features_part_50.csv            # Processed features (part 50)
│   └── final_features.csv                    # Complete processed dataset
├── src/                
|   ├── Preprocessed_Feature_engineering.ipynb    # Phase 1: Feature engineering pipeline
|   ├── Model_trainning.ipynb                     # Phase 2: Model training & evaluation
└── README.md                                 # Project documentation
```

## 📊 Dataset

- **Source**: Quora Question Pairs dataset
- **Size**: 50,000 samples (sampled from larger dataset)
- **Features**: After preprocessing, the dataset contains 6,000+ engineered features
- **Target**: Binary classification (0: Not duplicate, 1: Duplicate)

## 🔧 Workflow

### Phase 1: Feature Engineering (`Preprocessed_Feature_engineering.ipynb`)

#### 1. Data Loading & Preprocessing
- Loads data from 5 CSV files (train_part_1.csv to train_part_5.csv)
- Samples 50,000 rows for computational efficiency
- Handles missing values and removes duplicate entries

#### 2. Text Preprocessing
```python
def preprocess(q):
    # Converts text to lowercase
    # Handles special characters (%, $, ₹, €, @)
    # Removes [math] tags
    # Expands contractions (can't → can not)
    # Removes HTML tags using BeautifulSoup
    # Removes punctuation and URLs
    # Applies Porter Stemming
```

#### 3. Feature Engineering

**Basic Features:**
- `q1_len`, `q2_len`: Character length of questions
- `q1_num_words`, `q2_num_words`: Word count
- `word_common`: Common words between questions
- `word_total`: Total unique words
- `word_share`: Ratio of common to total words

**Token Features:**
- `cwc_min/max`: Common word count (min/max normalization)
- `csc_min/max`: Common stop word count
- `ctc_min/max`: Common token count
- `first_word_eq`, `last_word_eq`: Binary features for word matching

**Length Features:**
- `abs_len_diff`: Absolute difference in word count
- `mean_len`: Average word count
- `longest_substr_ratio`: Longest common substring ratio

**Fuzzy Features:**
- `fuzz_ratio`: Basic fuzzy string matching
- `fuzz_partial_ratio`: Partial string matching
- `token_sort_ratio`: Token-sorted matching
- `token_set_ratio`: Token-set matching
- `word_lcs_ratio`: Longest common subsequence of words
- `word_edit_similarity`: Word-level edit distance

#### 4. TF-IDF Vectorization
- Combines all questions into corpus
- Creates TF-IDF vectors with max_features=3000
- Splits vectors back to Q1 and Q2 representations
- Final dataset: **6,000+ features**

**Output**: `final_features.csv` (uploaded to GitHub)

---

### Phase 2: Model Training (`Model_trainning.ipynb`)

#### 1. Data Loading
- Loads preprocessed features from 50 CSV files (final_features_part_1.csv to final_features_part_50.csv)
- Concatenates all parts for comprehensive training dataset

#### 2. Memory-Efficient Training
```python
def run_classification_experiments(df, target_column, scale_features=True, sample_size=5000):
    # Samples data to fit memory constraints
    # Applies StandardScaler for feature scaling
    # Trains models sequentially to optimize memory usage
```

#### 3. Model Evaluation Pipeline
```python
def nlp_scorer(model_name, model, preprocessor, X, y):
    # 5-fold cross-validation
    # Multiple metrics: Accuracy, F1, ROC-AUC
    # Train-test split for additional validation
    # Training time measurement
```

#### 4. Model Ensemble
- **Random Forest**: Tree-based ensemble
- **Extra Trees**: Randomized tree ensemble
- **Gradient Boosting**: Sequential boosting
- **AdaBoost**: Adaptive boosting
- **XGBoost**: Extreme gradient boosting

#### 5. Hyperparameter Optimization
```python
def optimize_random_forest(X, y, n_trials=100):
    # Uses Optuna for Bayesian optimization
    # Optimizes: n_estimators, max_depth, min_samples_split, etc.
    # Cross-validation based selection
```

## 📈 Results

Sample results from 5,000 samples:

| Model | F1 Score | Accuracy | 
|-------|----------|----------|
| AdaBoost | 0.037 | 0.712 |
| Gradient Boosting | 0.021 | 0.745 | 
| XGBoost | 0.020 | 0.742 |
| Random Forest | 0.017 | 0.751 |
| Extra Trees | 0.013 | 0.758 |

## 🛠 Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting
- **NLTK**: Natural language processing
- **BeautifulSoup**: HTML parsing
- **FuzzyWuzzy**: Fuzzy string matching
- **Optuna**: Hyperparameter optimization
- **Distance**: String distance metrics

## 🚦 How to Run

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost nltk beautifulsoup4 fuzzywuzzy optuna distance
```

### Step 1: Feature Engineering
1. Run `Preprocessed_Feature_engineering.ipynb`
2. This generates `final_features.csv`
3. Split and upload to GitHub (if large)

### Step 2: Model Training
1. Update data loading URLs in `Model_trainning.ipynb`
2. Run the notebook
3. Models train sequentially to optimize memory usage

## ⚙ Configuration Options

### Memory Management
- `sample_size`: Adjust based on available RAM (default: 5,000)
- `n_estimators`: Reduce for faster training (default: 50)
- `cv_folds`: Reduce for memory efficiency (default: 5)

### Hyperparameter Tuning
- `n_trials`: Number of Optuna trials (default: 100)
- `timeout`: Maximum optimization time
- `n_jobs`: Parallel processing (-1 for all cores)

## 🎯 Key Features

- **Memory Efficient**: Sequential training and garbage collection
- **Comprehensive Features**: 6,000+ engineered features
- **Multiple Models**: Ensemble of 5 different algorithms
- **Hyperparameter Optimization**: Automated using Optuna
- **Robust Evaluation**: Cross-validation + train-test split
- **Scalable**: Handles large datasets through sampling

## 📝 Future Improvements

- [ ] Implement deep learning models (BERT, RoBERTa)
- [ ] Add more sophisticated feature selection
- [ ] Ensemble model combination techniques
- [ ] Real-time prediction API
- [ ] Web interface for question similarity checking
