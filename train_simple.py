#!/usr/bin/env python3
"""
Simple training script for sign language model
Uses existing training data from training_data directory
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("Training sign language model...")
    
    # Get training data path
    training_data_path = os.path.join(os.path.dirname(__file__), 'sign-language-main', 'training_data')
    
    X = []
    y = []
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    # Load all training data
    for letter in labels:
        letter_path = os.path.join(training_data_path, f'{letter}.npy')
        if os.path.exists(letter_path):
            samples = np.load(letter_path)
            X.extend(samples)
            y.extend([letter] * len(samples))
            print(f"Loaded {len(samples)} samples for letter '{letter}'")
    
    if len(X) == 0:
        print("No training data found!")
        return
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total samples: {len(X)}")
    print(f"Shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training accuracy: {accuracy:.2%}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'sign-language-main', 'model_nn.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
