from bs4 import BeautifulSoup
import pandas as pd



years = list(range(1977,2023))
dfs = []
for year in years:
    with open("awards/{}.html".format(year),encoding="utf-8") as f: 
        page = f.read()
    soup = BeautifulSoup(page,"html.parser")
    soup.find('tr',class_="over_header").decompose()
    roy_data = soup.find(id="roy")

    roy_table = pd.read_html(str(roy_data))[0]
    roy_table["Year"] = year
    dfs.append(roy_table)
    
roy_data = pd.concat(dfs)

roy_data.to_csv("roy_1976on.csv",index=False)

