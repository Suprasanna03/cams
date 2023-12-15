import json
from urllib.request import urlopen

url = "http://ipinfo.io/json"
res = urlopen(url)
data = json.load(res)

print(data)
