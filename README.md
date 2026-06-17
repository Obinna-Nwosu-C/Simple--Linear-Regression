Flood Probability Prediction in Nigeria using Simple Linear Regression

Project Overview
This project explores the application of Simple Linear Regression to predict the likelihood of flooding in Nigeria. Using a comprehensive environmental dataset, the model attempts to predict a continuous `FloodProbability` target variable based on a single environmental feature: `MonsoonIntensity`. 
The goal of this project is to demonstrate the end-to-end machine learning workflow—including data loading, exploratory data analysis (EDA), train/test splitting, model training, and performance evaluation—while highlighting the limitations of univariate (single-feature) linear models.

The Dataset
The analysis utilizes the Flood Prediction Dataset (sourced via Kaggle/Google Drive), containing 50,000 records and  21 environmental and infrastructural features. 

Target Variable: `FloodProbability` (Continuous decimal ranging from 0.0 to 1.0)
Selected Feature for Simple Regression:  `MonsoonIntensity` (Integer scale)
Note: While the dataset contains other 20 potential predictor variables (such as ClimateChange, Deforestation, and TopographyDrainage), Simple Linear Regression restricts the model to evaluating only one feature at a time.

Methodology
The project follows the standard CRISP-DM (Cross-Industry Standard Process for Data Mining) / Machine Learning lifecycle using Python:
1. Data Ingestion & EDA: Loaded the 50,000-row dataset using `pandas`. Visualized feature distributions using histograms and mapped the linear relationship between Monsoon Intensity and Flood Probability via scatter plots.
2. Train/Test Splitting: To evaluate model generalization, the data was split into training and testing sets. Two experiments were conducted to test model stability:
   Experiment A: 80% Training / 20% Testing
   Experiment B: 20% Training / 80% Testing
3. Model Training: Implemented `scikit-learn`'s `LinearRegression` (Ordinary Least Squares) to find the line of best fit.
4. Evaluation: Assessed model performance using standard regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).

Key Findings & Results
The Regression Equation (80/20 Split)
The OLS algorithm determined the following line of best fit:
Flood Probability = 0.4750 + (0.0049 × Monsoon Intensity)

Performance Metrics
| Metric | 80% Train / 20% Test | 20% Train / 80% Test |
|R-Squared (R2)| 0.06 | 0.05 |
|RMSE| 0.05 | 0.05 |
|MAE| 0.04 | 0.04 |

Analytical Insights
Positive Correlation: There is a definitive positive relationship; as monsoon intensity increases, so does the probability of flooding.
The Limitation of Simple Regression: The R2 score of ~0.06 indicates that `MonsoonIntensity` alone only explains roughly 6% of the variance in flood probability. 
Model Stability: The near-identical metrics between the 80/20 and 20/80 splits prove the model is stable and not overfitting, but fundamentally constrained by having only one input feature.
Conclusion:
Flooding is a highly complex, multivariate event. Accurately predicting it requires factoring in the other 19 variables present in the dataset (such as topography, infrastructure, and deforestation).

Technologies Used
Python 3.14.5
Pandas: Data manipulation and statistical summarization.
NumPy: Array reshaping and numerical operations.
Scikit-Learn:  Model building (`Linear Regression`), train/test splitting, and evaluation metrics.
Matplotlib: Data visualization (scatter plots, histograms, and regression lines).


How to Run
bash
# Clone the repository
git clone https://github.com/Obinna-Nwosu-C/Simple--Linear-Regression.git
cd Simple--Linear-Regression

# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Run the scripts (Python Code & Python Notebook)
Simple-Linear-Regression-Flood1.ipynb. The result of running this script is Simple-Linear-Regression-Flood2.ipynb.

Results
The scatter plots and evaluation results for the predictions with different dataset split ratios are in this repository.

Key Takeaways
    Linear regression is a powerful yet interpretable baseline model for continuous prediction tasks.
    The quality of the input feature directly impacts model performance — stronger correlations yield lower error metrics.
    Train/test splitting is essential to verify that a model generalizes to new, unseen data rather than simply memorizing the training set.
    The outcome of this prediction also proves that a "Simple" Linear Regression (using only one clue) is too simple for a complex real-world     problem like flooding.


Clear Interpretation of Your Results

1. The Mathematical Equation
From your 80% Train / 20% Test model, the algorithm found this line of best fit:
Flood Probability = 0.4750 + (0.0049 × Monsoon Intensity)
The Intercept (0.4750): If there is absolutely zero monsoon intensity, the baseline probability of a flood is still roughly 47.5%. (This makes sense, as floods can be caused by other things like dam failures or poor drainage).
The Slope (0.0049): For every 1-unit increase in Monsoon Intensity, the Flood Probability increases by 0.0049 (or about 0.5%).

2. The R-Squared Score
Your R2-score is 0.06 (for the 80/20 split) and 0.05 (for the 20/80 split). 
What this means: Monsoon Intensity alone only explains about 5% to 6% of the reasons why floods happen in this dataset. 
The takeaway: The other 94% of the flood probability is driven by the other 19 features in your dataset (like Climate Change, Deforestation, Topography Drainage, etc.) or random noise. 
Note: This doesn't mean that the code is wrong! The code is perfect. It just proves that a "Simple" Linear Regression (using only one clue) is too simple for a complex real-world problem like flooding.

3. Model Stability (80/20 vs. 20/80)
On testing the model with only 20% of the data for training, 
The slope dropped slightly from 0.0049 to 0.0048.
The R2 dropped slightly from 0.06 to 0.05.
Takeaway: The metrics are incredibly stable. This proves your model isn't overfitting; it's just consistently capturing that weak 5% linear relationship regardless of how much data you feed it.

Future Work
To capture the remaining 94% of unexplained variance, the next phase of this project will involve:
Multiple Linear Regression: Incorporating all 21 environmental features to see how they collectively impact flood probability.
Feature Selection: Using correlation matrices to identify and drop multicollinear features.
Advanced Algorithms: Testing non-linear models like Random Forests or XGBoost to capture complex, non-linear interactions between climate and infrastructure variables.

Created by Obinna Nwosu C.
