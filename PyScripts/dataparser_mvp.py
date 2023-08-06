from bs4 import BeautifulSoup
import pandas as pd



years = list(range(1977,2023))
dfs = []
for year in years:
    with open("awards/{}.html".format(year),encoding="utf-8") as f: 
        page = f.read()
    soup = BeautifulSoup(page,"html.parser")
    soup.find('tr',class_="over_header").decompose()
    mvp_data = soup.find(id="mvp")

    mvp_table = pd.read_html(str(mvp_data))[0]
    mvp_table["Year"] = year
    dfs.append(mvp_table)
    
mvp_data = pd.concat(dfs)

mvp_data.to_csv("mvp_1976on.csv",index=False)


