# Walmart-Sales-Prediction

## 1.Objective

The goal of this project is to build a regression model that can predict the purchase amount of customers based on their demographic and product-related information using a real-world dataset from Walmart.

## Dataset
from kaggle:walmart
 <a href=" https://github.com/Aleenapatani03/Walmart-Sales-Prediction/blob/main/walmart.csv ">Dataset</a>
## 2.Tools & Technologies

Python
Pandas, NumPy
Matplotlib
Seaborn
Scikit-learn

## 3. Data Preprocessing

- Loaded the dataset and checked for nulls.

- Removed unnecessary columns (like user/product IDs).

- Converted categorical variables to numerical using Label Encoding.

- Removed outliers from Purchase using IQR.

- No scaling was applied to Purchase since tree models don't require it.
Proper encoding and cleaning help models learn patterns better and prevent bias or error from skewed or missing data.

## 4. Exploratory Data Analysis (EDA)

- Distribution of Purchase values.
- Purchase amount vs. Age, Gender, Marital Status.
- Product category wise purchases.
- City Category vs. average Purchase.
  
EDA helped us understand which features are most related to the purchase amount and gave insights for feature engineering.

##  5. Model Building
Models Used:
- RandomForestRegressor
- DecisionTreeRegressor
- XGBRegressor

Why These Models?
- All are tree-based ensemble models:
- Handle non-linearities well.
- Do not require scaling.
- Work well with label-encoded features.

## 6. Model Training & Evaluation
Metrics Used:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- R² Score (Higher is better)
Each model was evaluated using these metrics.

## 7. Model Comparison Results
- Random Forest Regressor
  R² Score: 0.6411
  MAE (Mean Absolute Error): 2236.9
  MSE (Mean Squared Error): 8.8 Million

- Decision Tree Regressor
  R² Score: 0.6343
  MAE: 2258.8
  MSE: 8.9 Million

- XGBoost Regressor
  R² Score: 0.6412
  MAE: 2233.8
  MSE: 8.8 Million
  
Conclusion: Both Random Forest and XGBoost performed similarly, with XGBoost having a slightly better MAE.

## 8. Visualization

- Used scatter plots to compare actual vs. predicted values for each model.
- A perfect model would align all points along a 45° line.
- Visuals show that predictions are generally in the right range, but there is spread due to variance in customer behavior.

## 9. Final Conclusion

In this project, we aimed to predict the purchase amount of customers based on demographic and product-related features using regression models. After performing data preprocessing, exploratory data analysis (EDA), encoding, and outlier removal, we trained and evaluated three regression models: Random Forest, Decision Tree, and XGBoost.

All three models showed decent performance with R² scores around 0.63–0.64, indicating they could explain ~64% of the variance in the purchase data. Among them, XGBoost Regressor delivered the best results with the highest R² score (0.6412) and the lowest MAE (2233.8), making it the most suitable model for this prediction task.

The XGBoost Regressor was the best-performing model in this project and is recommended as the final model for deployment or further improvement.

##  10. Project Summary

This project shows how to preprocess real-world retail data, choose the right models, and evaluate model performance effectively.

Tools Used: Python, Pandas, Scikit-learn, XGBoost, Seaborn, Matplotlib.

ML Techniques: Regression, Ensemble Learning, Evaluation Metrics, Outlier Handling.




