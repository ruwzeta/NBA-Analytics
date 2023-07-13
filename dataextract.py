import pandas as pd
import requests
import time
from selenium import webdriver
years = list(range(2006,2023))

url_s = "https://www.basketball-reference.com/awards/awards_{}.html"

for year in years:
    url = url_s.format(year)
    data = requests.get(url)
    with open("awards/{}.html".format(year),"w+",encoding="utf-8") as f:
        f.write(data.text)


driver = webdriver.Chrome(executable_path = "D:\Assignments\Python Portfolios\NBA Analytics")
 