import requests
from bs4 import BeautifulSoup
response = requests.get('https://www.imdb.com/title/tt0145487')
# print(response.content)
#print(response.text)

# soup = BeautifulSoup(response.text, 'html.parser')
soup = BeautifulSoup(response.text, 'lxml')
# print(soup.find(class_='cast_list'))
print(soup.find(text='Budget:').parent.parent)
budget = soup.find(text='Budget:').parent.parent
print(budget)
budget_val = budget.find('h4').next_sibling
b = ''
for n in budget_val:
    if(n.isdigit()):
        b = b + n

print(b)
# table = soup.find_all(class_='cast_list')
# print(table.find_all('a'))

    def getHeraldArticles(self):
        article = ''
        
        # 1. Get all links in an object
        links = self.getCorpusLinks(self.herald_tag)

        # Go over links and write out articles
        for url in links:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'lxml')
                articles_div = soup.find('article',attrs={'class':'w112'})

                # For some reason this kills it
                for para in articles_div.findAll('p'):
                    article += ''.join(para.findAll(text=True)) + ' '
                
                # Write articles to corpus
                self.write_to_file(article, self.herald_tag)
            except AttributeError:
                print('Failed to find what we looked for')
                print('Sleeping to stop IP block...')
                time.sleep(10)