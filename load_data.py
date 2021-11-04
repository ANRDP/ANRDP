from os import getcwd
import pandas as pd
import io
import requests
import pathlib

destination = getcwd()
filename = destination + "/heart.csv"
file = pathlib.Path(filename)
if not file.exists():
    url = "https://raw.githubusercontent.com/ANRDP/ANRDP/main/heart.csv"
    data = requests.get(url)
    dataframe = pd.read_csv(io.StringIO(data.content.decode('utf-8')))
    dataframe.to_csv(filename)
else:
    dataframe = pd.read_csv(filename)
    print(dataframe.head)