import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from perceptron import Perceptron

# Define paths and URLs for the dataset
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'iris.data')

def load_dataset():
    """
    Loads the Iris dataset from a local directory or downloads it if unavailable.
    """
    if not os.path.exists(DATA_PATH):
        print("Downloading the Iris dataset...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        # The dataset does not have a header row, so we treat the first row as data
        df = pd.read_csv(DATA_URL, header=None)
        df.to_csv(DATA_PATH, index=False, header=False)
    else:
        print("Loading local Iris dataset...")
        df = pd.read_csv(DATA_PATH, header=None)
    return df

def main():
    # 1. Load and prepare the data
    df = load_dataset()
    df = shuffle(df, random_state=42)

    # Extract features and convert explicitly to standard NumPy arrays to prevent indexing errors
    X = df.iloc[:, 0:4].to_numpy(dtype=float)
    y = df.iloc[:, 4].to_numpy()

    # Split the dataset: 75% for training and 25% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Task 3: Encode the labels to 1 (Iris-setosa) and 0 (Other)
    y_train = np.where(y_train == 'Iris-setosa', 1, 0)
    y_test = np.where(y_test == 'Iris-setosa', 1, 0)

    # 2. Train the Perceptron model
    print("\nTraining the Custom Perceptron model...")
    model = Perceptron(eta=0.1, n_iter=10)
    model.fit(X_train, y_train)

    # Evaluate the model's accuracy
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    print(f"Model Training Accuracy: {round(accuracy, 2) * 100}%\n")

    # 3. Task 1: Manual Feature Input and Prediction
    print("-" * 40)
    print("Manual Iris-setosa Predictor")
    print("-" * 40)
    
    try:
        sepal_length = float(input("Enter sepal length (e.g., 5.1): "))
        sepal_width = float(input("Enter sepal width (e.g., 3.5): "))
        petal_length = float(input("Enter petal length (e.g., 1.4): "))
        petal_width = float(input("Enter petal width (e.g., 0.2): "))
        
        # Format the inputs for the model
        manual_features = np.array([sepal_length, sepal_width, petal_length, petal_width])
        
        # Generate the prediction
        prediction = model.predict(manual_features)

        # Output the formal result
        if prediction == 1:
            print("\nPrediction Result: The provided measurements correspond to 'Iris-setosa'.")
        else:
            print("\nPrediction Result: The provided measurements DO NOT correspond to 'Iris-setosa'.")
            
    except ValueError:
        print("\nError: Please ensure you enter valid numerical values for all features.")

if __name__ == "__main__":
    main()