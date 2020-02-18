import threading
from queue import Queue
import requests
from bs4 import BeautifulSoup
import csv

q = Queue()
lock = threading.Lock()
run = True

with open('/Users/Povilas/Desktop/Final-Year-Project/datasets/movies_without_budget.csv', mode='r',  encoding="utf8") as cmdb:
        movies = csv.reader(cmdb, delimiter=";", quotechar='"')

        with open('/Users/Povilas/Desktop/Final-Year-Project/datasets/movies_with_scrapped_budget.csv', 'w', newline='') as csvfile:
            fw = csv.writer(csvfile, delimiter=',',quotechar='"')

            # q = Queue()
            # lock = threading.Lock()

            def getBudgetFromIMDB(movieID):
                try:
                    url = 'https://www.imdb.com/title/' + movieID
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'lxml')
                    budget_html = soup.find(text='Budget:').parent.parent
                    budget_rawval = budget_html.find('h4').next_sibling
                    budget = ''
                    for n in budget_rawval:
                        if(n.isdigit()):
                            budget = budget + n
                    with lock:
                        fw.writerow([movieID, budget])
                        print('Added budget for ' + movieID)

                except AttributeError:
                    #print("No budget found for " + movieID)
                    pass


            def threader():
                while run:
                    movieID = q.get()
                    getBudgetFromIMDB(movieID)
                    q.task_done()

            for x in range(256):
                t = threading.Thread(target = threader)
                t.daemon = True
                t.start()

            for row in movies:
                q.put(row[0])
                #print(row[0])

            q.join()
            run = False

print('done')

