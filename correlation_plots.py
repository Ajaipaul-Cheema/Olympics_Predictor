import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('cleaned_data.csv')
output_folder = 'correlation_plots'
os.makedirs(output_folder, exist_ok=True)

features = ['Athletes', 'avg_years_of_schooling', 'GDP_per_capita_USD', 'Life Expectancy at Birth', 'Population']
filenames = ['athletes_vs_medals.png', 'avg_years_of_schooling_vs_medals.png', 'gdp_per_capita_vs_medals.png', 'life_expectancy_vs_medals.png', 'population_vs_medals.png']

for feature, filename in zip(features, filenames):
    # inspired by: https://datagy.io/python-seaborn-scatterplot/
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature, y='Medals')
    plt.title(f'Correlation between {feature} and Total Medals')
    plt.xlabel(feature)
    plt.ylabel('Total Medals')
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

print("Correlations plots saved successfully.")