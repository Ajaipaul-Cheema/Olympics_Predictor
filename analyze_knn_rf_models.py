import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

output_folder = 'compare_rf_knn_plots'
os.makedirs(output_folder, exist_ok=True)

with open('rfModel.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('knnModel.pkl', 'rb') as f:
    knn_model = pickle.load(f)

df = pd.read_csv('cleaned_data.csv')

# Predict 2016

# get around copy-slice warning
predictions = df.copy()
predictions = predictions[predictions['Year']==2016]

X = predictions[['Athletes', 'avg_years_of_schooling', 'GDP_per_capita_USD', 'Life Expectancy at Birth', 'Population']]
y = predictions['Medals']

rf_predictions = rf_model.predict(X)
knn_predictions = knn_model.predict(X)

predictions['RF_Predictions'] = rf_predictions
predictions['KNN_Predictions'] = knn_predictions

predictions['RF_Residuals'] = predictions['Medals'] - predictions['RF_Predictions']
predictions['KNN_Residuals'] = predictions['Medals'] - predictions['KNN_Predictions']

# Get residuals and outliers
std_rf = predictions['RF_Residuals'].std()
std_knn = predictions['KNN_Residuals'].std()

outlier_threshold_rf = 2.5 * std_rf
outlier_threshold_knn = 2.5 * std_knn
outliers_rf = predictions[(abs(predictions['RF_Residuals']) > outlier_threshold_rf)]
outliers_knn = predictions[(abs(predictions['KNN_Residuals']) > outlier_threshold_knn)]

std_rf = predictions['RF_Residuals'].std()
std_knn = predictions['KNN_Residuals'].std()

outlier_threshold_rf = 2.5 * std_rf
outlier_threshold_knn = 2.5 * std_knn
outliers_rf = predictions[(abs(predictions['RF_Residuals']) > outlier_threshold_rf)]
outliers_knn = predictions[(abs(predictions['KNN_Residuals']) > outlier_threshold_knn)]

plt.figure(figsize=(12, 6))
plt.scatter(predictions['Medals'], predictions['RF_Predictions'], label='RF Predictions', alpha=0.5)
plt.scatter(predictions['Medals'], predictions['KNN_Predictions'], label='KNN Predictions', alpha=0.5)
max_medals = max(predictions['Medals'].max(), predictions['RF_Predictions'].max(), predictions['KNN_Predictions'].max())
plt.plot([0, max_medals], [0, max_medals], color='red', linestyle='--')

for i, row in outliers_rf.iterrows():
    plt.text(row['Medals'], row['RF_Predictions'], f'{row["Country"]}', fontsize=8, ha='left', color='blue')

for i, row in outliers_knn.iterrows():
    plt.text(row['Medals'], row['KNN_Predictions'], f'{row["Country"]}', fontsize=8, ha='left', color='orange')

plt.xlabel('Actual Medals')
plt.ylabel('Predicted Medals')
plt.title('Actual vs Predicted Medals for All Countries (2016)')
plt.legend()
plt.savefig(os.path.join(output_folder,'actual_vs_predicted_medals.png'))

plt.figure(figsize=(12, 6))
plt.scatter(x='RF_Predictions', y='RF_Residuals', data=predictions, label='RF Residuals', alpha=0.5)
plt.scatter(x='KNN_Predictions', y='KNN_Residuals', data=predictions, label='KNN Residuals', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')

for i, row in outliers_rf.iterrows():
    plt.text(row['RF_Predictions'], row['RF_Residuals'], f'{row["Country"]}', fontsize=8, ha='right', color='blue')

for i, row in outliers_knn.iterrows():
    plt.text(row['KNN_Predictions'], row['KNN_Residuals'], f'{row["Country"]}', fontsize=8, ha='right', color='orange')

plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals (2016, Random Forest & KNN)')
plt.legend()
plt.savefig(os.path.join(output_folder,'residuals_comparison_rf_knn.png'))

predictions[['Country', 'Medals','RF_Predictions', 'KNN_Predictions','KNN_Residuals','RF_Residuals']].to_csv('knn_rf_predictions_2016.csv', index=False)
