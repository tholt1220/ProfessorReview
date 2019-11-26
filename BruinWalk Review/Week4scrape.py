import bs4 
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup 

m_url = "http://menu.dining.ucla.edu/Menus/Rendezvous"

uClient = uReq(m_url) #download 
m_html = uClient.read() #dump content into variable
uClient.close()

my_page = soup(m_html, 'html.parser')


print(my_page.h2)

print(my_page.findAll('div', {'class' : 'menu-item'}))