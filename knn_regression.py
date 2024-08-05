import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

df = pd.read_csv('cleaned_data.csv')
output_folder = 'knn_plots'
os.makedirs(output_folder, exist_ok=True)

X = df[['Athletes', 'avg_years_of_schooling', 'GDP_per_capita_USD', 'Life Expectancy at Birth', 'Population']]
y = df['Medals']
years = df['Year']

train_test_mask = years < 2016
predict_mask = years == 2016

X_train_test, y_train_test = X[train_test_mask], y[train_test_mask]
X_predict, y_predict = X[predict_mask], y[predict_mask]

X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.25, random_state=38)

pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(3))

"""
# inspired by: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# this was used to find the best param: 3 neighbours
param_grid = {'kneighborsregressor__n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)

best_knn = grid_search.best_estimator_
"""
pipeline.fit(X_train, y_train)

y_pred_knn = pipeline.predict(X_predict)

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# inspired by: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
mse_knn = mean_squared_error(y_predict, y_pred_knn)
rmse_knn = rmse_scorer(y_predict, y_pred_knn)  
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
# inspired by:  https://scikit-learn.org/stable/modules/cross_validation.html
cv_scores_knn = cross_val_score(pipeline, X_train_test, y_train_test, cv=5, scoring='neg_mean_squared_error')

# print("KNN Best Params:", grid_search.best_params_)
print("KNN Mean Squared Error:", mse_knn)
print(f"KNN RMSE (2016): {rmse_knn}")
print(f'KNN Training Score (R^2): {train_score}')
print(f'KNN Testing Score (R^2): {test_score}')
print("KNN Cross-Validation Mean Squared Error Scores:", cv_scores_knn)
print("KNN Cross-Validation Mean Mean Squared Error Score:", cv_scores_knn.mean())

correlation_matrix = df[['Athletes', 'avg_years_of_schooling', 'GDP_per_capita_USD', 'Life Expectancy at Birth', 'Population', 'Medals']].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.savefig(os.path.join('correlation_plots', 'Heatmap_Features_Correlation_Matrix.png'))

results = pd.DataFrame({
    'Train R2': [train_score],
    'Test R2': [test_score],
    'RMSE': [rmse_knn]  
})

results.to_csv('knn_results.csv', index=False)

with open('knnModel.pkl','wb') as f:
    pickle.dump(pipeline,f)