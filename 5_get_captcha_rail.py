from bs4 import BeautifulSoup
import time
import urllib.request
import requests

count = 0

session = requests.Session()
while (count<=1000):
    count += 1
    response = session.get('https://irs.thsrc.com.tw/IMINT/', cookies={'from-my': 'browser'})
    source = response.content.decode('utf-8')
    soup = BeautifulSoup(source, 'html.parser')
    img = soup.find('img')
    url = 'https://irs.thsrc.com.tw'+img.get('src')
    print(url)
    response = session.get(url, cookies={'from-my': 'browser'})
    with open('image/'+str(count)+'.png', 'wb') as file:
        file.write(response.content)
        file.flush()
