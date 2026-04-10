import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
import pickle

# Load dataset
print("Loading dataset...")
df = pd.read_csv('laptop_data.csv')

# ADVANCED CLEANING
print("Cleaning data...")
df['ram_gb'] = df['ram_gb'].str.extract('(\d+)').astype(int)
df['ssd'] = df['ssd'].str.extract('(\d+)').astype(int)
df['hdd'] = df['hdd'].str.extract('(\d+)').astype(int)
df['hdd'] = (df['hdd'] / 1024).astype(int)
df['graphic_card_gb'] = df['graphic_card_gb'].str.extract('(\d+)').astype(int)
df['warranty'] = df['warranty'].str.replace(' year', '').str.replace('s', '').replace('No warranty', '0').astype(int)

# Clean processor_gnrtn: map to integers
map_gnrtn = {'10th': 10, '11th': 11, '12th': 12, '7th': 7, '8th': 8, '9th': 9, '4th': 4, 'Not Available': 0}
df['processor_gnrtn'] = df['processor_gnrtn'].map(map_gnrtn)

# Features
features = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 
            'ram_type', 'ssd', 'hdd', 'os', 'graphic_card_gb', 
            'weight', 'warranty', 'Touchscreen', 'msoffice']
target = 'Price'

df = df[features + [target]].dropna()

X = df.drop(target, axis=1)
y = df[target]

categorical_features = ['brand', 'processor_brand', 'processor_name', 'ram_type', 'os', 'weight', 'Touchscreen', 'msoffice']
numerical_features = ['ram_gb', 'ssd', 'hdd', 'graphic_card_gb', 'warranty', 'processor_gnrtn']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
])

# Gradient Boosting often performs better on this scale of data
regressor = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

pipeline = TransformedTargetRegressor(
    regressor=Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ]),
    func=np.log1p,
    inverse_func=np.expm1
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"Training on {len(X_train)} samples...")
pipeline.fit(X_train, y_train)

train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"\n[DONE] Model Training Complete!")
print(f"Training R2 Score: {train_score:.4f}")
print(f"Testing R2 Score: {test_score:.4f}")

with open('model_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("\n[SUCCESS] Model saved as 'model_pipeline.pkl'")
