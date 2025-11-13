import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
print("Loading dataset...")
df = pd.read_csv('laptop_data.csv')  # ← CHANGED from 'laptopPrice.csv'

print(f"Dataset loaded: {len(df)} rows")

# DATA CLEANING - Remove 'GB' and convert to integers
print("Cleaning data...")

# Clean ram_gb: "8 GB" -> 8
df['ram_gb'] = df['ram_gb'].str.replace(' GB', '').astype(int)

# Clean ssd: "512 GB" -> 512
df['ssd'] = df['ssd'].str.replace(' GB', '').astype(int)

# Clean hdd: "1024 GB" -> 1 (convert to TB)
df['hdd'] = df['hdd'].str.replace(' GB', '').astype(int)
df['hdd'] = (df['hdd'] / 1024).astype(int)  # Convert GB to TB

# Clean graphic_card_gb: "4 GB" -> 4
df['graphic_card_gb'] = df['graphic_card_gb'].str.replace(' GB', '').astype(int)

# Clean warranty: "1 year" -> 1, "No warranty" -> 0
df['warranty'] = df['warranty'].str.replace(' year', '').str.replace('s', '')
df['warranty'] = df['warranty'].replace('No warranty', '0').astype(int)

print("Data cleaning complete!")

# Select relevant columns
selected_columns = ['brand', 'processor_brand', 'processor_name', 'ram_gb', 
                   'ram_type', 'ssd', 'hdd', 'os', 'graphic_card_gb', 
                   'weight', 'warranty', 'Touchscreen', 'msoffice', 'Price']

df = df[selected_columns]
df = df.dropna()

print(f"Final dataset: {len(df)} rows")

# Separate features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Define categorical columns for encoding
categorical_cols = ['brand', 'processor_brand', 'processor_name', 'ram_type', 
                   'os', 'weight', 'Touchscreen', 'msoffice']
categorical_indices = [0, 1, 2, 4, 7, 9, 11, 12]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), categorical_indices)
    ],
    remainder='passthrough'
)

# Create full pipeline
model_pipeline = Pipeline(steps=[
    ('transformation', preprocessor),
    ('model', RandomForestRegressor(max_depth=15, max_samples=0.85, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# Train model
print("\nTraining model...")
model_pipeline.fit(X_train, y_train)

# Evaluate
train_score = model_pipeline.score(X_train, y_train)
test_score = model_pipeline.score(X_test, y_test)

print(f"\n✓ Model Training Complete!")
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

# Save model
with open('model_pipeline.pkl', 'wb') as file:
    pickle.dump(model_pipeline, file)

print("\n✓ Model saved as 'model_pipeline.pkl'")
print("You can now run 'python app.py' to start the API server!")
