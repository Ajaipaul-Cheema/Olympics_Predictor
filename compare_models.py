import pandas as pd
import matplotlib.pyplot as plt

linear_results = pd.read_csv('linear_regression_results.csv')
knn_results = pd.read_csv('knn_results.csv')
rf_results = pd.read_csv('random_forest_results.csv')

linear_train_r2 = linear_results['Train R2'].iloc[0]
linear_test_r2 = linear_results['Test R2'].iloc[0]
linear_rmse = linear_results['RMSE'].iloc[0]

knn_train_r2 = knn_results['Train R2'].iloc[0]
knn_test_r2 = knn_results['Test R2'].iloc[0]
knn_rmse = knn_results['RMSE'].iloc[0]

rf_train_r2 = rf_results['Train R2'].iloc[0]
rf_test_r2 = rf_results['Test R2'].iloc[0]
rf_rmse = rf_results['RMSE'].iloc[0]

print("Linear Regression:")
print(f"Train R^2 Score: {linear_train_r2}")
print(f"Test R^2 Score: {linear_test_r2}")
print(f"RMSE: {linear_rmse}")

print("\nK-Nearest Neighbors:")
print(f"Train R^2 Score: {knn_train_r2}")
print(f"Test R^2 Score: {knn_test_r2}")
print(f"RMSE: {knn_rmse}")

print("\nRandom Forest:")
print(f"Train R^2 Score: {rf_train_r2}")
print(f"Test R^2 Score: {rf_test_r2}")
print(f"RMSE: {rf_rmse}")

results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'K-Nearest Neighbors', 'Random Forest'],
    'Train R^2 Score': [linear_train_r2, knn_train_r2, rf_train_r2],
    'Test R^2 Score': [linear_test_r2, knn_test_r2, rf_test_r2],
    'RMSE': [linear_rmse, knn_rmse, rf_rmse]
})

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].bar(results_df['Model'], results_df['Train R^2 Score'], color=['blue', 'orange', 'green'])
ax[0].set_title('Train R^2 Score Comparison')
ax[0].set_ylabel('Train R^2 Score')

ax[1].bar(results_df['Model'], results_df['Test R^2 Score'], color=['blue', 'orange', 'green'])
ax[1].set_title('Test R^2 Score Comparison')
ax[1].set_ylabel('Test R^2 Score')

ax[2].bar(results_df['Model'], results_df['RMSE'], color=['blue', 'orange', 'green'])
ax[2].set_title('2016 prediction RMSE Comparison')
ax[2].set_ylabel('RMSE')

plt.tight_layout()
plt.savefig('Model_Comparison.png')