# Resume Screening and Job Role Classification

## Project Overview

This project focuses on automatically classifying resumes into job roles using Machine Learning techniques. The goal is to assist recruiters by reducing manual effort in screening resumes and quickly identifying suitable candidates based on their skills and experience.

The model processes resume text, extracts meaningful features, and predicts the most relevant job category.

---

## Objective

* Build a robust text classification system for resumes
* Perform preprocessing on a messy real-world dataset
* Apply feature extraction and machine learning techniques
* Evaluate model performance using standard metrics

---

## Dataset

* Source: Kaggle Resume Dataset
* Link: https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset

The dataset contains structured resume fields such as:

* Skills
* Responsibilities
* Education details
* Job role labels

---

## Methodology

### 1. Data Preprocessing

* Removed hidden characters from column names
* Handled missing values using `fillna()`
* Combined important text fields (skills, responsibilities, education)
* Cleaned text (lowercasing, removing punctuation & numbers)
* Removed duplicate entries to avoid data leakage

### 2. Feature Engineering

* Converted text into numerical form using **TF-IDF Vectorization**
* Removed very common and very rare words to reduce noise
* Used n-grams (unigrams + bigrams) for better context

### 3. Label Simplification

* Grouped similar job roles into broader categories:

  * Engineer
  * HR
  * Database
  * Other (removed for better model performance)

### 4. Model Building

* Algorithm used: **Logistic Regression**
* Used **Pipeline** to combine preprocessing and model
* Applied **class balancing** to handle imbalanced data

### 5. Hyperparameter Tuning

* Used **GridSearchCV**
* Tuned regularization parameter `C`

---

## Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics help assess the model’s performance across different job categories.

---

## Results

* The model is able to classify resumes into job roles based on textual features
* Performance is influenced by overlapping vocabulary across job roles
* TF-IDF captures frequency-based patterns but has limitations in semantic understanding

---

## Limitations

* Resume data contains overlapping and generic terms across roles
* TF-IDF does not capture contextual meaning of words
* Some predictions may be biased toward frequently occurring patterns

---

## Future Improvements

* Use advanced NLP models like BERT for better semantic understanding
* Improve dataset quality and labeling
* Add more structured features (experience level, certifications)

---

## Model Saving

The trained model is saved as:

```
model/logistic_model.pkl
```

It can be loaded later for predictions without retraining.

---

## How to Run

1. Clone the repository
2. Install required libraries:

   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the Jupyter Notebook or Python script
4. Enter resume skills when prompted

---

## Sample Input

```
python machine learning data analysis
```

## Sample Output

```
Engineer
```

---

## Conclusion

This project demonstrates how machine learning can be applied to automate resume screening. While the model performs reasonably well, further improvements can be made using advanced NLP techniques.

---