import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Load the dataset
#current_dir = os.path.dirname(os.path.abspath(__file__))
#csv_file_path = os.path.join(current_dir, 'Flask', 'concrete_data.csv')  # Use raw string if preferred
data = pd.read_csv('Flask\concrete_data.csv')

# Function to remove outliers using the IQR method
def remove_outliers(df, columns, z_thresh=3):
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        z_score = (df[column] - mean) / std
        df = df[(z_score < z_thresh) & (z_score > -z_thresh)]
    return df

# Columns to remove outliers from
columns_to_clean = ['concrete_compressive_strength', 'water', 'blast_furnace_slag', 'superplasticizer', 'age', 'fine_aggregate ']

# Remove outliers
df_cleaned = remove_outliers(data, columns_to_clean)

# Define features and target
X = df_cleaned.drop(columns=['concrete_compressive_strength'])
y = df_cleaned['concrete_compressive_strength']

# Split the data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the scaler and model
#scaler_path = os.path.join(current_dir, 'Flask', 'scaler.pkl')
#model_path = os.path.join(current_dir, 'Flask', 'cement.pkl')

try:
    scaler = pickle.load(open('Flask\scaler.pkl', 'rb'))
    model = pickle.load(open('Flask\cement.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    raise

# Scale the test features
X_test_scaled = scaler.transform(X_test)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Optionally, save the predictions and actual values to a CSV file for further analysis
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv(os.path.join('Flask', 'model_predictions.csv'), index=False)
print("Predictions saved to Flask/model_predictions.csv")


