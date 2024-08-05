from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

df = pd.read_csv('cleaned_data.csv')
output_folder = 'feature_importance_plot'
os.makedirs(output_folder, exist_ok=True)

features = ['Athletes', 'avg_years_of_schooling', 'GDP_per_capita_USD', 'Life Expectancy at Birth', 'Population']

X = df[features]
y = df['Medals']

rf_model = RandomForestRegressor(n_estimators=100, random_state=38)
rf_model.fit(X, y)

feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# inspired by: https://datagy.io/python-seaborn-scatterplot/
plt.figure(figsize=(17, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values(by='Importance', ascending=False))
plt.title('Feature Importance for Random Forest')
plt.savefig(os.path.join(output_folder, 'feature_importance.png'))

print("Feature importance plots saved successfully.")