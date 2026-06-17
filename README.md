Simple Linear Regression: Predicting CO2 Emissions from Engine Size

Project Overview
This project demonstrates the fundamentals of Simple Linear Regression using Python and the scikit-learn library. The goal is to build a predictive model that estimates a vehicle's CO2 emissions (g/km) based solely on its engine size (liters).
Using the Ordinary Least Squares (OLS) method, the model fits a straight line through the data to find the mathematical relationship between these two variables, then evaluates how well that line generalizes to unseen data.

Dataset
The analysis uses the FuelConsumption.csv dataset from the Government of Canada, containing model-specific fuel consumption ratings and estimated CO2 emissions for new light-duty vehicles.
    Source: Open Government of Canada
    Input Feature (X): Engine Size (L)
    Target Variable (y): CO2 Emissions (g/km)
    Sample Size: ~1,068 vehicles

Methodology
The project follows the standard machine learning workflow:
    Data Loading & Exploration – Using Pandas to inspect statistics and Matplotlib to visualize relationships via histograms and scatter plots.
    Train/Test Split – The dataset is divided into 80% training data and 20% testing data using train_test_split to evaluate generalization.
    Model Training – A LinearRegression model is fitted using the OLS algorithm to find the optimal slope and intercept.
    Prediction & Evaluation – The trained model predicts CO2 emissions on the unseen test set, and performance is measured using standard regression metrics.

Key Outputs
    Regression Equation: CO2 = Intercept + Coefficient × Engine Size
    Visualizations: Scatter plots with the fitted regression line for both training and test data
    Evaluation Metrics:
        Mean Absolute Error (MAE)
        Mean Squared Error (MSE)
        Root Mean Squared Error (RMSE)
        R² Score (coefficient of determination)

Technologies Used
    Python 3.14.5
    NumPy – Numerical computations
    Pandas – Data manipulation
    Matplotlib – Data visualization
    Scikit-learn – Machine learning model and evaluation metrics

How to Run
bash
# Clone the repository
git clone https://github.com/Obinna-Nwosu-C/Simple--Linear-Regression.git
cd Simple--Linear-Regression

# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Run the scripts (Python Code & Python Notebook)
Simple-Linear-Regression-v1.ipynb and python regression.py

Key Takeaways
    Linear regression is a powerful yet interpretable baseline model for continuous prediction tasks.
    The quality of the input feature directly impacts model performance — stronger correlations yield lower error metrics.
    Train/test splitting is essential to verify that a model generalizes to new, unseen data rather than simply memorizing the training set.
