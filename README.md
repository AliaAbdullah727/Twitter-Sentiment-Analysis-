# Sentiment Analysis of Tweets

## Project Overview
This project focuses on performing sentiment analysis on a large dataset of tweets. The goal is to classify tweets as either positive (1) or negative (0) using machine learning techniques. We utilize text preprocessing steps such as stemming and TF-IDF vectorization, followed by training a Logistic Regression model.

## Dataset
The dataset used for this project is the **Sentiment140 dataset**, obtained from Kaggle.

*   **Source:** [Sentiment140 Dataset](https://www.kaggle.com/datasets/milobele/sentiment140-dataset-1600000-tweets)
*   **Description:** Contains 1.6 million tweets extracted using the Twitter API. The tweets have been annotated for sentiment (0 = negative, 4 = positive). For this project, the '4' (positive) sentiment is converted to '1'.

## Dependencies
To run this notebook, you will need the following Python libraries:

*   `pandas`
*   `numpy`
*   `nltk`
*   `scikit-learn`
*   `opendatasets`
*   `joblib`
*   `torch` (for device detection, though not directly used in the model training here)

You can install most of these using pip:
```bash
pip install pandas numpy nltk scikit-learn opendatasets joblib
```

## Methodology
1.  **Data Download:** The dataset is downloaded directly from Kaggle using the `opendatasets` library.
2.  **Data Loading & Preprocessing:**
    *   The raw CSV is loaded into a pandas DataFrame, assigning appropriate column names.
    *   Missing values are checked (none found in this dataset).
    *   The target variable (sentiment) is mapped from `0` and `4` to `0` and `1` respectively.
3.  **Text Cleaning & Stemming:**
    *   A `stemming` function is defined to clean tweet text by removing non-alphabetic characters, converting to lowercase, splitting words, removing English stopwords, and applying Porter Stemming.
    *   This function is applied to the 'text' column to create a new 'stemmed_content' column.
4.  **Feature Extraction:**
    *   The `stemmed_content` is converted into numerical features using `TfidfVectorizer`.
5.  **Data Splitting:** The dataset is split into training and testing sets (80% training, 20% testing).
6.  **Model Training:** A `LogisticRegression` model is trained on the TF-IDF vectorized training data.
7.  **Model Evaluation:** The model's performance is evaluated using `accuracy_score` and `classification_report` on the test set.
8.  **Model Persistence:** The trained model is saved using `joblib` for future use and then reloaded to demonstrate its functionality.
9.  **Prediction on Sample:** The reloaded model is tested on a sample tweet from the dataset to verify its predictions.

## Results
The Logistic Regression model achieved a high accuracy on the test set:

*   **Accuracy:** 0.9884 (98.84%)
*   **Classification Report:**
    ```
                  precision    recall  f1-score   support

               0       1.00      0.98      0.99    160000
               1       0.98      1.00      0.99    160000

        accuracy                           0.99    320000
       macro avg       0.99      0.99      0.99    320000
    weighted avg       0.99      0.99      0.99    320000
    ```

## Usage
To replicate this project:

1.  **Clone this repository** (or download the notebook).
2.  **Install the dependencies** listed above.
3.  **Run the Jupyter Notebook/Colab Notebook** cells in sequence.
4.  You will be prompted to enter your Kaggle credentials to download the dataset during the `opendatasets` step. If you don't have them, you can create an account on Kaggle and generate an API token.

The trained model will be saved as `logistic_regression_model.pkl` in the current directory.
