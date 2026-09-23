Markdown
# Custom Perceptron Implementation - Week 01

## Overview
This repository contains a from-scratch implementation of a custom Perceptron classifier used to classify the Iris dataset. The primary objective of this assignment was to understand the underlying mechanics of a Perceptron by manually coding its weight updates, activation function, and prediction logic, and then extending its capabilities based on specific assigned tasks.

## Step-by-Step Assignment Implementation

### 1. Project Modularization
Instead of writing a single script, the project is divided into a professional directory structure:
*   `src/perceptron.py`: Contains the core algorithm logic.
*   `src/evaluate.py`: Handles data loading, preprocessing, training, and user interaction.
*   `data/raw/`: Excluded from version control, used to store the downloaded dataset locally.

### 2. Modifying the Core Algorithm (`perceptron.py`)
The base Perceptron algorithm calculates the weighted sum of inputs and a bias term to make predictions. To fulfill **Task 2**, the standard step function was replaced with a Sigmoid activation function.   
*   **Weighted Sum:** Computes the dot product of the input features and weights, adding the bias term.   
*   **Sigmoid Activation:** Returns a continuous value between 0 and 1.   
*   **Prediction Threshold:** A threshold of 0.5 is applied to the Sigmoid output to map the continuous result back to binary class predictions (1 or 0).
*   **Weight Updating:** During the `.fit()` process, the algorithm iterates through the data `n_iter` times, updating weights using the formula: `weight += learning_rate * (target - predicted) * input_feature`.   

### 3. Data Preprocessing & Label Encoding (`evaluate.py`)
The model uses the UCI Iris dataset. The data processing pipeline performs the following:   
*   **Data Loading:** Automatically downloads `iris.data` if it is not found locally.   
*   **Shuffling & Splitting:** Shuffles the data to randomize the order and splits it 75% for training and 25% for testing. Explicitly converts Pandas DataFrames to NumPy arrays (`.to_numpy()`) to prevent PyArrow indexing conflicts with Scikit-learn.   
*   **Label Encoding (Task 3):** The original target labels (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) were transformed. `Iris-setosa` was mapped to 1, and all other species were mapped to 0, replacing the standard 1 and -1 perceptron labels.   

### 4. Interactive Manual Predictor (Task 1)
To fulfill **Task 1**, an interactive command-line interface was added to `evaluate.py`.   
*   After the model trains and reports its accuracy on the test set, it prompts the user to manually enter four feature values: sepal length, sepal width, petal length, and petal width.   
*   These inputs are converted into a NumPy array and passed directly into the trained model's `.predict()` method.
*   The model outputs a real-time assessment declaring whether the manually entered measurements correspond to an `Iris-setosa` or not.   

## How Task 2 and Task 3 Were Accomplished

### Task 2: Replace the Step Function with Sigmoid
The original tutorial used a simple Step Function that returned 1 if the weighted sum was greater than 0, and -1 otherwise. In `src/perceptron.py`, the step function was replaced with these two methods:   

```python
    def sigmoid(self, X):
        # Calculates the mathematical Sigmoid curve (outputs between 0.0 and 1.0)
        return 1.0 / (1.0 + np.exp(-self.weighted_sum(X)))

    def predict(self, X):
        # Uses the Sigmoid output and applies a 0.5 threshold to determine the class
        return np.where(self.sigmoid(X) >= 0.5, 1, 0)
This forces the network to calculate a probability (via Sigmoid) rather than a rigid step, successfully fulfilling Task 2.

Task 3: Replace the labels of 1, -1 with 1 & 0
The original tutorial encoded the Iris-setosa labels as 1 and all other flowers as -1. Since the Sigmoid function outputs values between 0 and 1, the labels need to match that scale. In src/evaluate.py, this was accomplished with the following lines:

Python
    # Task 3: Encode the labels to 1 (Iris-setosa) and 0 (Other)
    y_train = np.where(y_train == 'Iris-setosa', 1, 0)
    y_test = np.where(y_test == 'Iris-setosa', 1, 0)
This ensures the target outputs perfectly align with the new Sigmoid activation function, successfully fulfilling Task 3.
How to Run the Project
1. Install Dependencies

Bash
pip install -r requirements.txt
2. Execute the Evaluation Script

Bash
python src/evaluate.py
This command will trigger the data download, train the model, display the test accuracy, and initiate the interactive manual predictor.