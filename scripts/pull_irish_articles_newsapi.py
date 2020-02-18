import requests
import os


# Tapping into RTE news and Irish times articles
# By default returns 10 results for TopArticles end point
# Need to use payed version to get full content of articles.

class newsAPI:
       
       def __init__(self, key):
              self.key = key
              self.rte_tag = 'RTE'
              self.irish_times_tag = 'IRISH-TIMES'
       
       def write_to_file(self, news_data, tag=None):
              DATA_PATH = "../corpus/irishArticles/"

              if tag == self.rte_tag:
                     DATA_PATH = DATA_PATH + self.rte_tag + '/' + 'Counter.txt'
              elif tag == self.irish_times_tag:
                     DATA_PATH =  DATA_PATH + self.irish_times_tag + '/' + 'Counter.txt'
              else:
                     print('No article path specified!')

              newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_PATH))

              with open(newspaper_list_file, 'w') as filehandle:
                     filehandle.write('%s\n' % news_data)
       
       def getAllSources(self):
              url = ('https://newsapi.org/v2/sources?'
                     'country=ie&'
                     'apiKey=a3257b78687444049b7c378559dc92e7')
              
              response = requests.get(url)
              print(response.json())

       def getTopIrishHeadLines(self):
              url = ('https://newsapi.org/v2/top-headlines?'
                     'country=ie&'
                     'apiKey=a3257b78687444049b7c378559dc92e7')
              
              response = requests.get(url)
              print(response.json())
       
       def getTopRTEArticles(self):
              # Cannot mix country and sources tag
              url = ('https://newsapi.org/v2/top-headlines?'
                     # 'country=ie&'
                     'sources=RTE&'
                     'apiKey=a3257b78687444049b7c378559dc92e7')
              
              response = requests.get(url)
              self.write_to_file(response.json(), self.rte_tag)

       def getTopIrishTimesArticles(self):
              url = ('https://newsapi.org/v2/top-headlines?'
                     # 'country=ie&'
                     'sources=the-irish-times&'
                     'apiKey=a3257b78687444049b7c378559dc92e7')
              
              response = requests.get(url)
              self.write_to_file(response.json(), self.irish_times_tag)
       

       def getRTEArticles(self):
              # Cannot mix country and sources tag
              url = ('https://newsapi.org/v2/everything?'
                     'from=2020-01-01&'
                     # 'to=2020-01-02&'
                     # 'country=ie&'
                     'sources=RTE&'
                     'apiKey=a3257b78687444049b7c378559dc92e7')
              
              response = requests.get(url)
              self.write_to_file(response.json(), self.rte_tag)

       def getIrishTimesArticles(self):
              url = ('https://newsapi.org/v2/everything?'
                     # 'country=ie&'
                     'sources=the-irish-times&'
                     'apiKey=a3257b78687444049b7c378559dc92e7')
              
              response = requests.get(url)
              self.write_to_file(response.json(), self.irish_times_tag)



def main():
       apiKey = 'a3257b78687444049b7c378559dc92e7'

       newsClient = newsAPI(apiKey)
       
       # newsClient.getAllSources()

       # newsClient.getTopIrishHeadLines()

       # newsClient.getTopRTEArticles()

       # newsClient.getTopIrishTimesArticles()

       newsClient.getRTEArticles()

       # newsClient.getIrishTimesArticles()


if __name__ == '__main__':
       main()

