import requests
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd

# Function to find table in a website, giving the URL and a distinct column name
def findTable(url,colFind):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table')
    target_table = None
    for table in tables:
        th_elements = table.find_all('th')
        if any(colFind in th.get_text() for th in th_elements):
                target_table = table
                break
    return pd.read_html(StringIO(str(target_table)))[0]

# Scrape olympic data from wikipedia for given years
# Obtains Year,Country,Medals,Total Athletes for each olympic
def scrapeWiki(years):
    all_data = []
    for y in years:
        print(f"Scraping data for {y}")
        url1 = f"https://en.wikipedia.org/wiki/{y}_Summer_Olympics"
        participants = findTable(url1,'IOC Letter Code')

        url2 = f"https://en.wikipedia.org/wiki/{y}_Summer_Olympics_medal_table"
        medals = findTable(url2,'Total')
        if 'NOC' in medals.columns:
            medals.rename(columns={'NOC': 'Nation'}, inplace=True)
        medals['Nation'] = medals['Nation'].str.replace('*', '', regex=False)
        medals['Nation'] = medals['Nation'].str.replace('‡', '', regex=False)
        full = participants.merge(medals, left_on='Country', right_on='Nation', how='left')
        full['Total'] = full['Total'].fillna(0).astype(int)
        full['Year'] = y
        all_data.append(full)
    result = pd.concat(all_data, ignore_index=True)
    result = result[['Year','Country','Total','Athletes']]
    result.rename(columns={'Total': 'Medals'}, inplace=True)
    result.to_csv('rawdata/olympic_wiki.csv', index=False)

def main():
        years = [1992, 1996, 2000, 2004, 2008, 2012, 2016]
        scrapeWiki(years)

if __name__ == '__main__':
     main()