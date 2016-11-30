from bs4 import BeautifulSoup
import urllib

thisurl = "https://en.wikipedia.org/wiki/Hewlett-Packard"
soup = BeautifulSoup(urllib.urlopen(thisurl).read(),'html.parser')

data=''.join([x.text for x in soup.find(id='mw-content-text').find_all('p')]).encode('utf-8');
with open("trainingdata.txt", "a") as myfile:
    myfile.write(data)
    myfile.close()