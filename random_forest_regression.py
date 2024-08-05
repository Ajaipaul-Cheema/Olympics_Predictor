import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv('cleaned_data.csv')
output_folder = 'random_forest_plots'
os.makedirs(output_folder, exist_ok=True)

X = df[['Athletes', 'avg_years_of_schooling', 'GDP_per_capita_USD', 'Life Expectancy at Birth', 'Population']]
y = df['Medals']

years = df['Year']  

train_test_mask = years < 2016
predict_mask = years == 2016

X_train_test, y_train_test = X[train_test_mask], y[train_test_mask]
X_predict, y_predict = X[predict_mask], y[predict_mask]

X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.25, random_state=38)


"""
# this was used to determine the right parameters for the random forest regressor model
# inspired by: https://stackoverflow.com/questions/53782169/random-forest-tuning-with-randomizedsearchcv
param_distributions = {
    'n_estimators': np.arange(50, 501, 50),
    'max_depth': np.arange(5, 21, 2),
    'min_samples_split': np.arange(2, 11, 1),
    'min_samples_leaf': np.arange(1, 11, 1),
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestRegressor(random_state=38)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=100, 
    cv=5,  
    scoring='accuracy',
    n_jobs=-1,  
    random_state=38,
    verbose=2
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best parameters:", best_params)

best_rf = random_search.best_estimator_
"""

pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(
    n_estimators=215,           
    max_depth=4,               
    min_samples_split=8,        
    min_samples_leaf=4,         
    max_features='sqrt',        
    random_state=38             
))

pipeline.fit(X_train, y_train)
y_pred_rf = pipeline.predict(X_predict)

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# inspired by: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
mse_rf = mean_squared_error(y_predict, y_pred_rf)
rmse_rf = rmse_scorer(y_predict, y_pred_rf)  
train_score_rf = pipeline.score(X_train, y_train)
test_score_rf = pipeline.score(X_test, y_test)

# inspired by: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# inspired by:  https://scikit-learn.org/stable/modules/cross_validation.html
cv_scores_rf = cross_val_score(pipeline, X_train_test, y_train_test, cv=5, scoring=mse_scorer)

print(f"Random Forest Mean Squared Error (2016): {mse_rf}")
print(f"Random Forest RMSE (2016): {rmse_rf}")
print(f"Random Forest Training Score (R^2): {train_score_rf}")
print(f"Random Forest Testing Score (R^2): {test_score_rf}")
print(f"Random Forest Cross-Validation Mean Squared Error Scores: {cv_scores_rf}")
print(f"Random Forest Cross-Validation Mean Mean Squared Error Score: {cv_scores_rf.mean()}")

results = pd.DataFrame({
    'Train R2': [train_score_rf],
    'Test R2': [test_score_rf],
    'RMSE': [rmse_rf]  
})

results.to_csv('random_forest_results.csv', index=False)

with open('rfModel.pkl','wb') as f:
    pickle.dump(pipeline,f)