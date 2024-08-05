import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

output_folder = 'compare_rf_knn_plots'
os.makedirs(output_folder, exist_ok=True)

with open('rfModel.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('knnModel.pkl', 'rb') as f:
    knn_model = pickle.load(f)

df = pd.read_csv('cleaned_data.csv')

canada_data = df[df['Country'] == 'Canada']

years = [1992, 1996, 2000, 2004, 2008, 2012, 2016]

X_canada = canada_data[['Athletes', 'avg_years_of_schooling', 'GDP_per_capita_USD', 'Life Expectancy at Birth', 'Population']]
y_canada = canada_data['Medals']

rf_test_score = rf_model.score(X_canada, y_canada)
knn_test_score = knn_model.score(X_canada, y_canada)

print(f"Random Forest Prediction Score: {rf_test_score}")
print(f"K-Nearest Neighbors Test Score: {knn_test_score}")

predictions_rf = rf_model.predict(X_canada)
predictions_knn = knn_model.predict(X_canada)

predictions_df = pd.DataFrame({
    'Year': years,
    'Actual Medals': y_canada.tolist(),
    'Predicted Medals RF': predictions_rf,
    'Predicted Medals KNN': predictions_knn
})

predictions_df.to_csv('canada_medal_predictions.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(predictions_df['Year'], predictions_df['Actual Medals'], marker='o', label='Actual Medals')
plt.plot(predictions_df['Year'], predictions_df['Predicted Medals RF'], marker='o', linestyle='--', label='Predicted Medals RF')
plt.plot(predictions_df['Year'], predictions_df['Predicted Medals KNN'], marker='o', linestyle='--', label='Predicted Medals KNN')
plt.xlabel('Year')
plt.ylabel('Medals')
plt.title('Canada Olympic Medal Predictions')
plt.legend()
plt.xticks(years, labels=years)
plt.savefig(os.path.join(output_folder,'canada_medal_predictions.png'))
