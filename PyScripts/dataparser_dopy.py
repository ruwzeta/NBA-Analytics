from bs4 import BeautifulSoup
import pandas as pd



years = list(range(2022,2023))
dfs = []
for year in years:
    with open("awards/{}.html".format(year),encoding="utf-8") as f: 
        page = f.read()
    soup = BeautifulSoup(page,"html.parser")
    soup.find('tr',class_="over_header").decompose()
    dpoy_data = soup.find(id="dpoy")

    dpoy_table = pd.read_html(str(dpoy_data))[0]
    dpoy_table["Year"] = year
    dfs.append(dpoy_table)
    
dpoy_data = pd.concat(dfs)

dpoy_data.to_csv("dpoy_1983on.csv",index=False)

