# Bagging-classifier-simulation-tool
This project demonstrates the implementation of a Bagging Classifier using different base estimators (Decision Tree, SVM, and KNN) in Python. The app is built using Streamlit, which provides an interactive web interface to visualize decision boundaries and evaluate model performance.

## Features
- **Base Estimator Selection**: The user has a choice between Decision Tree, SVM, and KNN as the base estimator.
- **Hyperparameter Tuning**:
  - Number of estimators
  - Maximum samples
  - Bootstrap samples
  - Maximum features
  - Bootstrap features
- **Accuracy Evaluation**: Calculates and displays the accuracy of the classifiers.
- **Visualization**: Decision boundaries for both the base estimator and the Bagging Classifier are displayed

## Dataset
The project uses a toy dataset created by combining two different datasets. One of the datasets is the "Social Network Ads" dataset available on Kaggle. You can find it here: [Social Network Ads Dataset](https://www.kaggle.com/datasets/d4rklucif3r/social-network-ads). This synthetic dataset simulates social network ads, with features like age, estimated salary, and purchased status.

## Prerequisites
To run this project, you need the following installed on your system:

- Python 3.7+
- Required Python libraries (install using `pip`):
  ```bash
  pip install numpy pandas seaborn streamlit scikit-learn matplotlib
  ```

## Steps to Run the Project
1. Clone the repository or download the script.
2. Place the `Social_Network_Ads.csv` file in the same directory as the script.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Interact with the sidebar to configure the Bagging Classifier settings.

## WebApp User Interface

### Sidebar Controls
- **Base Estimator**: Choose between Decision Tree, SVM, and KNN.
- **Number of Estimators**: Specify the number of base models in the ensemble.
- **Max Samples**: Adjust the number of samples to train each base model.
- **Bootstrap Samples**: Enable or disable sampling with replacement.
- **Max Features**: Specify the maximum number of features for training each base model.
- **Bootstrap Features**: Enable or disable feature sampling with replacement.

### Outputs
- **Decision Boundary Visualization**: Displays decision boundaries for the selected base estimator and the Bagging Classifier.
- **Accuracy**: Shows the accuracy of both classifiers on the test dataset.
- **Sample Data**: Displays a random sample of 5 rows from the dataset that were used in the process.

## Code Overview

### Import Libraries
The code uses:
- `numpy` and `pandas` for data manipulation
- `seaborn` and `matplotlib` for visualization
- `scikit-learn` for machine learning models and preprocessing
- `streamlit` for building the web app

### Preprocessing
- Splits the dataset into training and testing sets.
- Standardizes the feature values using `StandardScaler`.

### Bagging Classifier
- Implements a Bagging Classifier with configurable parameters through the Streamlit interface.
- Trains both the base estimator and the Bagging Classifier to visualize and compare decision boundaries.

## Acknowledgments
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
