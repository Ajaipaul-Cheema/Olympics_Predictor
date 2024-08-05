import pandas as pd
import numpy as np

years = [1992, 1996, 2000, 2004, 2008, 2012, 2016]
def filterYears(df):
    return df[df['Year'].isin(years)]

# Load the datasets
population_data = pd.read_csv('rawdata/world_population.csv')
life_expectancy_data = pd.read_csv('rawdata/life_expectancy_at_birth.csv')
schooling_data = pd.read_csv('rawdata/mean_years_of_schooling.csv', delimiter=';')
gdp_data = pd.read_csv('rawdata/world_country_gdp.csv')
olympic_data = pd.read_csv('rawdata/olympic_wiki.csv')

# Rename columns
population_data = population_data.rename(columns={"country": "Country"})
gdp_data = gdp_data.rename(columns={"Country Name": "Country"})
gdp_data = gdp_data.rename(columns={"year": "Year"})
schooling_data = schooling_data.rename(columns={"Entity": "Country"})

# recover 100 rows by fixing country names
olympic_data['Country'] = olympic_data['Country'].replace('Czech Republic', 'Czechia')
olympic_data['Country'] = olympic_data['Country'].replace('Great Britain', 'United Kingdom')
olympic_data['Country'] = olympic_data['Country'].replace('Macedonia', 'North Macedonia')
olympic_data['Country'] = olympic_data['Country'].replace('São Tomé and Príncipe', 'Sao Tome and Principe')
olympic_data['Country'] = olympic_data['Country'].replace('Swaziland', 'Eswatini')
olympic_data['Country'] = olympic_data['Country'].replace('East Timor', 'Timor-Leste')
olympic_data['Country'] = olympic_data['Country'].replace('Republic of the Congo', 'Congo')
olympic_data['Country'] = olympic_data['Country'].replace('Zaire', 'Democratic Republic of the Congo')
gdp_data['Country'] = gdp_data['Country'].replace('Czech Republic', 'Czechia')
gdp_data['Country'] = gdp_data['Country'].replace("Hong Kong SAR, China", 'Hong Kong')
gdp_data['Country'] = gdp_data['Country'].replace("Korea, Rep.", 'South Korea')
gdp_data['Country'] = gdp_data['Country'].replace("Bahamas, The", 'Bahamas')
gdp_data['Country'] = gdp_data['Country'].replace("Turkiye", 'Turkey')
gdp_data['Country'] = gdp_data['Country'].replace("Russian Federation", 'Russia')
gdp_data['Country'] = gdp_data['Country'].replace("St. Lucia", 'Saint Lucia')
gdp_data['Country'] = gdp_data['Country'].replace("Congo, Rep.", 'Congo')
life_expectancy_data['Country'] = life_expectancy_data['Country'].replace("Russian Federation", 'Russia')
gdp_data['Country'] = gdp_data['Country'].replace("Syrian Arab Republic", 'Syria')
gdp_data['Country'] = gdp_data['Country'].replace("Egypt, Arab Rep.", 'Egypt')
gdp_data['Country'] = gdp_data['Country'].replace("Iran, Islamic Rep.", 'Iran')
gdp_data['Country'] = gdp_data['Country'].replace("Slovak Republic", 'Slovakia')
gdp_data['Country'] = gdp_data['Country'].replace("Venezuela, RB", 'Venezuela')
gdp_data['Country'] = gdp_data['Country'].replace("Kyrgyz Republic", 'Kyrgyzstan')
gdp_data['Country'] = gdp_data['Country'].replace("Congo, Dem. Rep.", 'Democratic Republic of the Congo')
life_expectancy_data['Country'] = life_expectancy_data['Country'].replace("Syrian Arab Republic", 'Syria')
life_expectancy_data['Country'] = life_expectancy_data['Country'].replace("Viet Nam", 'Vietnam')
life_expectancy_data['Country'] = life_expectancy_data['Country'].replace("The Democratic Republic of the Congo", 'Democratic Republic of the Congo')
population_data['Country'] = population_data['Country'].replace("Czech Republic (Czechia)", 'Czechia')
population_data['Country'] = population_data['Country'].replace("Côte d'Ivoire", 'Ivory Coast')
gdp_data['Country'] = gdp_data['Country'].replace("Cote d'Ivoire", 'Ivory Coast')
schooling_data['Country'] = schooling_data['Country'].replace("Cote d'Ivoire", 'Ivory Coast')
schooling_data['Country'] = schooling_data['Country'].replace("Timor", 'Timor-Leste')
schooling_data['Country'] = schooling_data['Country'].replace("Democratic Republic of Congo", 'Democratic Republic of the Congo')

# Pivot columns longer
life_expectancy_data = life_expectancy_data.drop(columns=['ISO3','Continent','Hemisphere','Human Development Groups','UNDP Developing Regions','HDI Rank (2021)'])
life_expectancy_data = pd.melt(life_expectancy_data, id_vars=["Country"], var_name="Year", value_name="Life Expectancy at Birth")
life_expectancy_data["Year"] = life_expectancy_data["Year"].str.extract(r'(\d{4})').astype(int)
life_expectancy_filtered = filterYears(life_expectancy_data)

population_data = pd.melt(population_data, id_vars=["Country"], var_name="Year", value_name="Population")
population_data["Year"] = population_data["Year"].str.extract(r'(\d{4})').astype(int)

# Filter out years
schooling_filtered = filterYears(schooling_data)
gdp_filtered = filterYears(gdp_data)
olympic_filtered = filterYears(olympic_data)
life_expectancy_filtered = filterYears(life_expectancy_data)
population_data_filtered = filterYears(population_data)

# Drop unwanted cols
gdp_filtered = gdp_filtered.drop(columns=['Country Code','GDP_USD'])
schooling_filtered = schooling_filtered.drop(columns = ['Code'])

# Merge datasets
merged_df = olympic_filtered.merge(schooling_filtered, on=["Country", "Year"]).merge(gdp_filtered, on=["Country", "Year"]).merge(life_expectancy_filtered, on=["Country", "Year"]).merge(population_data_filtered, on=["Country", "Year"])

print(merged_df.isnull().sum())
df_cleaned = merged_df.dropna()
print(len(df_cleaned))

df_cleaned.to_csv('cleaned_data.csv',index=False)