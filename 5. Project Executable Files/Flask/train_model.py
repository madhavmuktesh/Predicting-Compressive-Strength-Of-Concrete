import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, 'concrete_data.csv')
df = pd.read_csv("Flask\concrete_data.csv")

# Function to remove outliers
def remove_outliers(df, columns, z_thresh=3):
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        z_score = (df[column] - mean) / std
        df = df[(z_score < z_thresh) & (z_score > -z_thresh)]
    return df

# Columns to remove outliers from
columns_to_clean = ['concrete_compressive_strength', 'water', 'blast_furnace_slag', 'superplasticizer', 'age', 'fine_aggregate ']
df_cleaned = remove_outliers(df, columns_to_clean)

# Splitting features and target variable
x = df_cleaned.drop(columns=['concrete_compressive_strength'])
y = df_cleaned['concrete_compressive_strength']

# Scaling features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Define parameter distribution
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(1, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'subsample': uniform(0.6, 0.4),
    'max_features': ['sqrt', 'log2', None]
}

# Perform Randomized Search
gb = GradientBoostingRegressor()
random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_dist, 
                                   n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(x_train, y_train)

# Get the best model
best_gb = random_search.best_estimator_

# Save the best model and the scaler
model_path = os.path.join(current_dir, 'cement.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(best_gb, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Model and scaler trained and saved to {model_path} and {scaler_path} respectively.")



