# PySpark Project Portfolio

Welcome to the PySpark Project Portfolio! This repository showcases three projects that demonstrate key data processing and machine learning skills using PySpark. Each project highlights different aspects of PySpark, including data handling, machine learning, recommendation systems, and text processing.

## Table of Contents
1. [Project 1: Bank Marketing Binary Classification](#bank-marketing-project)
2. [Project 2: Movie Recommendation System](#movie-recommendation)
3. [Project 3: TF-IDF Text Processing with Gettysburg Address](#simple_search_engine)

---

## Project 1: Bank Marketing Binary Classification
**Objective**: Predict whether a client will subscribe to a term deposit based on demographic and campaign-related data.

- **Dataset**: Bank Marketing dataset from the UCI Machine Learning Repository.
- **Description**: This project uses a binary classification model to predict customer responses. The dataset includes categorical features like job, marital status, and education, which are processed with encoding techniques to make them suitable for machine learning models.
- **Model**: Decision Tree and Logistic Regression models, evaluated with metrics like AUC to gauge performance.
- **Key Skills**: Data preprocessing, categorical encoding, model training, and evaluation.

**Code Highlights**:
- Data transformation using `StringIndexer` and `OneHotEncoder` for categorical features.
- Training and evaluating models with PySpark’s MLlib.

**Run Instructions**:
1. Load the dataset in CSV format.
2. Run the code provided to preprocess, train, and evaluate the model.

---

## Project 2: Movie Recommendation System
**Objective**: Recommend movies to users based on historical ratings.

- **Dataset**: MovieLens 1M dataset, containing user ratings, movies, and user demographics.
- **Description**: A collaborative filtering recommendation system built with the Alternating Least Squares (ALS) algorithm. The model provides personalized movie recommendations for each user by learning latent factors from the user-movie interaction matrix.
- **Model**: ALS (Alternating Least Squares) for collaborative filtering.
- **Key Skills**: Collaborative filtering, model evaluation, data joining, and recommendations generation.

**Code Highlights**:
- Uses `ALS` for collaborative filtering on a large dataset.
- Top-N recommendations for each user, joined with movie titles for interpretability.

**Run Instructions**:
1. Place the MovieLens `.dat` files (ratings, movies, users) in the project directory.
2. Run the code to train the ALS model and generate top movie recommendations for each user.
3. Recommendations are saved to a CSV file for easy review.

---

## Project 3: TF-IDF Text Processing with Gettysburg Address
**Objective**: Identify the most relevant document based on a target word using TF-IDF.

- **Dataset**: Text excerpts from the Gettysburg Address and similar documents.
- **Description**: A text-processing project that uses Term Frequency-Inverse Document Frequency (TF-IDF) to rank document relevance based on a target word. This approach showcases PySpark’s handling of text data and feature extraction for document relevance.
- **Method**: TF-IDF for calculating document relevance scores.
- **Key Skills**: Text processing, feature extraction, relevance ranking.

**Code Highlights**:
- Implements TF-IDF using PySpark’s `HashingTF` and `IDF` classes.
- Ranks documents by relevance based on a target word.

**Run Instructions**:
1. Place text files in the specified directory.
2. Run the TF-IDF script with a target word to find the most relevant document.

---

## Conclusion
These projects collectively demonstrate the versatility of PySpark for various data science tasks, from machine learning and recommendation systems to text processing. Each project is designed to highlight PySpark’s capabilities in handling, transforming, and modeling data at scale.

**Feel free to explore each project, and if you have any questions, please reach out!**

---
