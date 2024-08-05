import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from sklearn.pipeline import make_pipeline

df = pd.read_csv('cleaned_data.csv')
output_folder = 'linear_regression_plots'
os.makedirs(output_folder, exist_ok=True)

X = df[['Athletes', 'avg_years_of_schooling', 'GDP_per_capita_USD', 'Life Expectancy at Birth', 'Population']]
y = df['Medals']
years = df['Year']

train_test_mask = years < 2016
predict_mask = years == 2016

X_train_test, y_train_test = X[train_test_mask], y[train_test_mask]
X_predict, y_predict = X[predict_mask], y[predict_mask]

X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.25, random_state=38)


pipeline = make_pipeline(StandardScaler(), LinearRegression())
pipeline.fit(X_train, y_train)

y_pred_lr = pipeline.predict(X_predict)

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# inspired by: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
mse_lr = mean_squared_error(y_predict, y_pred_lr)
rmse_lr = rmse_scorer(y_predict, y_pred_lr)  
train_score_lr = pipeline.score(X_train, y_train)
test_score_lr = pipeline.score(X_test, y_test)

# inspired by: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# inspired by:  https://scikit-learn.org/stable/modules/cross_validation.html
cv_scores_lr = cross_val_score(pipeline, X_train_test, y_train_test, cv=5, scoring=mse_scorer)

print(f"Linear Regression Mean Squared Error (2016): {mse_lr}")
print(f"Linear Regression RMSE (2016): {rmse_lr}")
print(f"Linear Regression Training Score (R^2): {train_score_lr}")
print(f"Linear Regression Testing Score (R^2): {test_score_lr}")
print(f"Linear Regression Cross-Validation Mean Squared Error Scores: {cv_scores_lr}")
print(f"Linear Regression Cross-Validation Mean Mean Squared Error Score: {cv_scores_lr.mean()}")

plt.figure(figsize=(12, 6))
plt.scatter(y_predict, y_pred_lr, color='blue', label='Predicted vs Actual')
plt.xlabel('Actual Medals')
plt.ylabel('Predicted Medals')
plt.title('Actual vs Predicted Medals (2016)')
plt.plot([y_predict.min(), y_predict.max()], [y_predict.min(), y_predict.max()], color='red', linestyle='--')
plt.legend()
plt.savefig(os.path.join(output_folder, 'Linear_Regression_Prediction_vs_Actual_2016.png'))

residuals = y_predict - y_pred_lr
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_lr, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Medals')
plt.ylabel('Residuals')
plt.title('Residuals Plot (2016)')
plt.savefig(os.path.join(output_folder, 'Linear_Regression_Residual_Plot_2016.png'))

results = pd.DataFrame({
    'Train R2': [train_score_lr],
    'Test R2': [test_score_lr],
    'RMSE': [rmse_lr]  
})

results.to_csv('linear_regression_results.csv', index=False)

predictions = pd.DataFrame({
    'Year': years[predict_mask],
    'Actual Medals': y_predict,
    'Predicted Medals': y_pred_lr
})

predictions.to_csv('linear_regression_predictions_2016.csv', index=False)