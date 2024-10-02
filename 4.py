from bs4 import BeautifulSoup
import selenium
import pandas
import requests
url='https://www.keellssuper.com'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    supnig=BeautifulSoup(response.content,'html.parser')
    bithtara=BeautifulSoup.find_all('div',class_='eggs')
    name_tag = bithtara.find('span', class_='biththara one')
    price_tag = bithtara.find('span', class_='i want bittara')
else:
    print(f"Failed to retrieve page. Status code: {response.status_code}")
