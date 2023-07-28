import pandas as pd
import requests
import time
years = list(range(1977,2006))

url_s = " https://www.basketball-reference.com/leagues/NBA_{}_per_game.html "

for year in years:
    url = url_s.format(year)
    
    data = requests.get(url)
    with open("players/{}.html".format(year),"w+",encoding="utf-8") as f:
        f.write(data.text)
    time.sleep(40)
