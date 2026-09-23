# Custom Perceptron Implementation

## Overview
This repository contains a custom implementation of a Perceptron classifier built from scratch using NumPy. It is designed to classify the Iris dataset, distinguishing the `Iris-setosa` species from others.

## Implementation Details
In accordance with the assignment requirements, the following modifications have been integrated:
- **Activation Function:** The standard step function has been replaced with a Sigmoid activation function.
- **Label Encoding:** Target labels have been standardized to `1` (Iris-setosa) and `0` (Other).
- **Interactive Evaluation:** A command-line interface allows users to input manual feature measurements for real-time inference.

## Repository Structure
- `data/raw/`: Directory for the unhandled Iris dataset.
- `notebooks/`: Contains Jupyter Notebooks for exploratory data analysis.
- `src/`: Houses the core source code (`perceptron.py`) and the evaluation script (`evaluate.py`).

## Usage Instructions
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt