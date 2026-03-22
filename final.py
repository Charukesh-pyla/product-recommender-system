import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

DB_PATH = 'reviews.db'
TABLE_NAME = 'amazon_reviews'
PLOT_DIR = 'plots'


os.makedirs(PLOT_DIR, exist_ok=True)

class RecommenderPipeline:
    def __init__(self, db_path=DB_PATH, table=TABLE_NAME,
                 min_reviews=3, svd_components=20, knn_neighbors=5, random_state=42):
        self.db_path = db_path
        self.table = table
        self.min_reviews = min_reviews
        self.svd_components = svd_components
        self.knn_neighbors = knn_neighbors
        self.random_state = random_state
        self.conn = None
        self.pivot = None
        self.product_ids = None
        self.knn_model = None
        self.svd = None
        self.X_pred = None
        self.rf_model = None
        self.X_rf_pred = None


    def import_csv_to_sqlite(self, csv_path):
        df = pd.read_csv(csv_path)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(subset=['user_id','product_id','rating'], inplace=True)
        self.conn = sqlite3.connect(self.db_path)
        df.to_sql(self.table, self.conn, if_exists='replace', index=False)
        logging.info(f"Imported CSV into SQLite table '{self.table}' ({len(df)} rows)")

        
        plt.figure(figsize=(6,4))
        sns.histplot(df['rating'], bins=10, kde=True)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.savefig(f"{PLOT_DIR}/rating_distribution.png")
        plt.close()

    def load_from_sql(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(f"SELECT * FROM {self.table}", self.conn)
        logging.info(f"Loaded {len(df)} rows from SQL table '{self.table}'")
        return df

    def perform_olap(self, df):
        user_counts = df['user_id'].value_counts()
        prod_counts = df['product_id'].value_counts()

        # Plots
        plt.figure(figsize=(6,4))
        sns.histplot(user_counts, bins=30, log_scale=(False, True))
        plt.title('User Review Counts')
        plt.xlabel('Reviews per User')
        plt.ylabel('Number of Users (log scale)')
        plt.savefig(f"{PLOT_DIR}/user_review_counts.png")
        plt.close()

        plt.figure(figsize=(6,4))
        sns.histplot(prod_counts, bins=30, log_scale=(False, True))
        plt.title('Product Review Counts')
        plt.xlabel('Reviews per Product')
        plt.ylabel('Number of Products (log scale)')
        plt.savefig(f"{PLOT_DIR}/product_review_counts.png")
        plt.close()

        df = df[df['user_id'].isin(user_counts[user_counts >= self.min_reviews].index)]
        df = df[df['product_id'].isin(prod_counts[prod_counts >= self.min_reviews].index)]
        return df

    def build_pivot(self, df):
        self.pivot = df.pivot_table(values='rating', index='user_id', columns='product_id', fill_value=0)
        self.product_ids = list(self.pivot.columns)
        logging.info(f"Pivot matrix shape: {self.pivot.shape}")

        # Plot heatmap (sampled for clarity)
        sample = self.pivot.iloc[:30, :30]
        plt.figure(figsize=(10,8))
        sns.heatmap(sample, cmap='YlGnBu')
        plt.title('Sample Pivot Table Heatmap (30x30)')
        plt.savefig(f"{PLOT_DIR}/pivot_heatmap.png")
        plt.close()

    def fit_models(self):
        data = self.pivot.values.T
        self.knn_model = NearestNeighbors(n_neighbors=self.knn_neighbors+1, metric='cosine').fit(data)
        logging.info("Trained KNN model.")

        self.svd = TruncatedSVD(n_components=self.svd_components, random_state=self.random_state)
        X_svd = self.svd.fit_transform(self.pivot.values)
        self.X_pred = self.svd.inverse_transform(X_svd)
        # Random Forest training
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        X_flat = self.pivot.values.reshape(-1, self.pivot.shape[1])
        y_flat = self.pivot.values.mean(axis=1)  # Simplified regression target
        self.rf_model.fit(X_flat, y_flat)
        self.X_rf_pred = np.tile(self.rf_model.predict(X_flat).reshape(-1, 1), (1, self.pivot.shape[1]))
        logging.info("Trained Random Forest model.")

        # Plot explained variance
        plt.figure(figsize=(6,4))
        plt.plot(np.cumsum(self.svd.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('SVD Explained Variance')
        plt.grid(True)
        plt.savefig(f"{PLOT_DIR}/svd_explained_variance.png")
        plt.close()
        logging.info("Trained SVD model and plotted explained variance.")

    def top_n_by_average(self, df, n=10):
        avg = df.groupby('product_id')['rating'].mean()
        top = avg.sort_values(ascending=False).head(n).index.tolist()
        return top

    def knn_recommend(self, product_id, top_n):
        if product_id not in self.product_ids:
            raise ValueError("Product ID not in pivot matrix.")
        idx = self.product_ids.index(product_id)
        distances, indices = self.knn_model.kneighbors(self.pivot.values.T[idx].reshape(1,-1), n_neighbors=top_n+1)
        rec_ids = [self.product_ids[i] for i in indices.flatten()[1:]]
        return rec_ids
    
    def random_forest_recommend(self, product_id, top_n):
        if product_id not in self.product_ids:
            raise ValueError("Product ID not in pivot matrix.")
        j = self.product_ids.index(product_id)
        # Use the predicted matrix (same rating per user, tiled across all items)
        product_scores = self.X_rf_pred[:, j]
        ranked_indices = np.argsort(product_scores)[::-1]
        recommended = [self.product_ids[i] for i in ranked_indices if self.product_ids[i] != product_id][:top_n]
        return recommended


    def svd_recommend(self, product_id, top_n):
        if product_id not in self.product_ids:
            raise ValueError("Product ID not in pivot matrix.")
        j = self.product_ids.index(product_id)
        sims = np.dot(self.X_pred.T, self.X_pred[:,j])
        best = np.argsort(sims)[::-1]
        rec = [self.product_ids[i] for i in best if self.product_ids[i] != product_id][:top_n]
        return rec

    def evaluate(self, test_size=0.2):
        X = self.pivot.values
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=self.random_state)

        # ---- Random Forest Evaluation ----
        X_flat_test = X_test.reshape(-1, X.shape[1])
        y_rf_true = X_flat_test.mean(axis=1)
        y_rf_pred = self.rf_model.predict(X_flat_test)

        rf_mse = mean_squared_error(y_rf_true, y_rf_pred)
        rf_rmse = np.sqrt(rf_mse)
        rf_r2 = r2_score(y_rf_true, y_rf_pred)

        # ---- Plot Evaluation Metrics ----
        plt.figure(figsize=(8, 4))
        metric_names = [ 'RF_MSE', 'RF_RMSE', 'RF_R2']
        metric_vals = [ rf_mse, rf_rmse, rf_r2]
        plt.bar(metric_names, metric_vals, color=['navy', 'blue', 'skyblue', 'orange', 'darkorange', 'gold'])
        plt.title('Evaluation Metrics: KNN vs Random Forest')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/evaluation_metrics_knn_rf.png")
        plt.close()

        return {
            'RF_MSE': rf_mse, 'RF_RMSE': rf_rmse, 'RF_R2': rf_r2
        }



def main():
    pipe = RecommenderPipeline()
    pipe.import_csv_to_sqlite('amazon.csv')
    df = pipe.load_from_sql()
    df = pipe.perform_olap(df)
    pipe.build_pivot(df)
    pipe.fit_models()

    top10 = pipe.top_n_by_average(df, n=10)
    print("Top 10 products by average rating:", top10)

    inp = input("Enter a product ID for recommendations: ")
    try:
        knn_recs = pipe.knn_recommend(inp, top_n=10)
        svd_recs = pipe.svd_recommend(inp, top_n=10)
        rf_recs = pipe.random_forest_recommend(inp, top_n=10)

        print("KNN Recommendations:", knn_recs)
        print("SVD Recommendations:", svd_recs)
        print("Random Forest Recommendations:", rf_recs)
    except ValueError as e:
        print(e)


    evals = pipe.evaluate()
    print("Evaluation metrics:", evals)

if __name__ == '__main__':
    main()
