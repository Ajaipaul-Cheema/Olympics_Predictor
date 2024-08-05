# Olympics Country Medal Predictor

## Usage
* pip install pandas numpy matplotlib urllib3 requests bs4 lxml seaborn scikit-learn
* python main.py

## Data Processing
Loading Datasets
The project loads various datasets related to country statistics and Olympic performance:

* Population Data: rawdata/world_population.csv
* Life Expectancy Data: rawdata/life_expectancy_at_birth.csv
* Mean Years of Schooling Data: rawdata/mean_years_of_schooling.csv
* GDP Data: rawdata/world_country_gdp.csv
* Olympic Medal Data: rawdata/olympic_wiki.csv (scraped from Wikipedia)

## Missing Values
* Checking for Missing Values: The datasets are checked for missing values.
* Cleaning Data: Rows with missing values are dropped to ensure data integrity.
* Country Name Fixes: Various country names are standardized across datasets to ensure consistency (e.g., 'Czech Republic' to 'Czechia').

## Data Merging
The cleaned datasets are merged into a single DataFrame for analysis:
* Merging Process: The datasets are merged on Country and Year columns to create a comprehensive dataset containing relevant statistics for each country during the specified Olympic years.
* Final Dataset: The merged dataset is saved as cleaned_data.csv.

## Analysis and Visualization
The analysis includes several steps to understand the data and prepare it for modeling:

* Total Medal Distribution: Analysis of the total number of medals won by each country.
* Country Statistics Relationships: Exploring relationships between country statistics (e.g., GDP, life expectancy) and medal counts.
* Actual vs Predicted Medal Counts: For Canada dataset and for all countries dataset
* Residual Plots: For both KNN and RF

## Training and Modeling
Model Training:
For all models: The models used StandardScaler and used data prior to 2016, splitting training data into 75% and testing data into 25%. 

* Linear Regression: Trained the model with StandardScaler 
* K-Nearest Neighbors (KNN): Trained the model with StandardScaler and used 3 neighbours as the parameter
* Random Forest (RF): Trained the model with StandardScaler and used n_estimators=215, max_depth=4, min_samples_split=8, min_samples_leaf=4, max_features='sqrt'

Model Evaluation:
* Performance Metrics: Evaluating model performance on a separate test set which was the entire 2016 Olympics dataset using MSE,
RMSE, R^2 score, cross validation MSE, train score, and test score.

* Comparison with Alternative Models:
KNN vs RF: Created scatter plots of predicted vs actual medals. Created residual plots. Recorded csv files for comparison.
Both RF and KNN performed well; however, KNN was a little bit better in prediction.

## Contributors
- Ajaipaul Cheema
- Matthew Hamilton