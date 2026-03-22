# Product Recommender System

A machine learning-based product recommendation system developed as part of the **Data Warehousing and Data Mining (AD211)** course. This project analyzes user-product rating data and recommends products using multiple recommendation approaches, including **K-Nearest Neighbors (KNN)**, **Singular Value Decomposition (SVD)**, and **Random Forest Regressor**.

## Project Overview

Recommender systems are widely used in e-commerce and digital platforms to help users discover relevant products based on their preferences. This project builds a recommendation pipeline that processes rating data, stores it in a relational database, applies preprocessing techniques, trains multiple machine learning models, and generates personalized product recommendations.

## Problem Statement

The objective is to develop a recommendation system that predicts a user's interest in a product by analyzing past user-product interaction data.

The system aims to:
- Filter users and products with insufficient data
- Compare multiple recommendation models
- Recommend top-N products for a given user
- Store and manage data efficiently using SQLite
- Visualize patterns and model behavior using plots and heatmaps

## Features

- Data preprocessing and cleaning
- Filtering sparse user-product interactions
- SQLite database integration
- User-item pivot table creation
- Recommendation using KNN, SVD, and Random Forest Regressor
- Model evaluation using RMSE and R² Score
- Data visualization using histograms, heatmaps, and variance plots
- Top-N product recommendation generation

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Surprise
- SQLite3
- Matplotlib
- Seaborn

## Project Structure
```
Product-Recommender-System/
│
├── final.py
├── requirements.txt
├── amazon.csv
├── reviews.db
├── plots/
└── README.md
```

## Methodology

1. **Data Collection and Storage** — Load the dataset from CSV and store it in SQLite
2. **Data Preprocessing** — Clean missing values, filter sparse data, build user-item matrix
3. **Exploratory Data Analysis** — Analyze rating distributions and visualize sparsity
4. **Model Building** — Apply KNN, SVD, and Random Forest
5. **Evaluation and Recommendation** — Compare models using RMSE and R², generate recommendations

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/product-recommender-system.git
cd product-recommender-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python final.py
```

## Dataset

The project uses an Amazon ratings dataset. Place the file in the root directory as:
```
amazon.csv
```

## Output

- Rating distribution plots
- Heatmaps of the user-item pivot table
- SVD explained variance plots
- KNN similarity-based recommendations
- Random Forest predicted recommendations
- Evaluation metrics (RMSE and R²)

## Results

- **KNN** — Similarity-based collaborative filtering
- **SVD** — Latent factor matrix factorization
- **Random Forest** — Supervised learning approach for rating prediction

## Future Improvements

- Build a hybrid recommendation system
- Add a web-based interface for user interaction
- Deploy as a web application
- Use larger and more diverse datasets

## License

This project is developed for academic purposes only.
