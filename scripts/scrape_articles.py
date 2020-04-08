import requests
import threading
from bs4 import BeautifulSoup
from queue import Queue
import os
import csv
import re
import time
import string
from builtins import any as b_any

# Scraping articles over 2019
class WebScrapeArticles:
       
    def __init__(self):
        # News Company Tags+
        self.new_york_times_tag = "NEW_YORK_TIMES"
        self.irish_times_tag = 'IRISH-TIMES'
        self.daily_mail_tag = "DAILY_MAIL"
        self.independent_tag = "INDEPENDENT"

        self.call_tag = ''
        self.pull_links = False

        self.q = Queue()
        self.lock = threading.Lock()
        self.run = True

        self.EXTENSION = "_LINKS.txt"
        self.refined_news_categories = ['brexit', 'climate.change', 'stocks'
            , 'sport.gaa' ,'sport.football'
            , 'trump', 'politics', 'medicine'
            , 'cars', 'middle.east', 'abortion'
            , 'christianity', 'drugs'
            , 'USA', 'china', 'business'
            , 'housing', 'online', 'food.reviews']

    # Scraping links
    # Use REST API standards and search between dates for months such as january - December 2019.
    # Each month is 200 articles (max per search) so 2400 total results per year, 9600 results for the 4 years.
    # About 30 minutes for 2400 results.
    def buildArticleLinks(self, newsCompany):
        links = []
        print('PROGRAM STARTING EXECUTION...')

        if newsCompany == self.daily_mail_tag:
            page_range = 12
            link_range = 11
        elif newsCompany ==  self.independent_tag:
            # Iterate 40 times which is 40 times 10 perage so 400 articles per topic. The problem is that some links or pages are dead, aim for 60.
            page_range = 61
            # Iterate 10 times, since, 10 pages per result
            link_range = 11
        else:
            print('No expected page range specified')
            return 'ERROR 404 ADD CODE FOR COMPANY SCRAPE'

        # Will be pulling links until the end of this method
        self.pull_links = True

        # Replace news_cats with refined_news_categories after specific categories added
        for topic in self.refined_news_categories:
            print('TOPIC ' + topic + ' HAS STARTED EXECUTING')

            # Separate out the dot for the query
            if '.' in topic:
                topic = ' '.join(topic.split('.'))

            for page_number in range(1, page_range):
                # for link_number in range (1, link_range):
                if newsCompany == self.independent_tag:
                    links = self.getIndependentArticleLinks(topic, page_number)
                elif newsCompany == self.daily_mail_tag:
                    links = self.getDailyMailArticleLinks(topic, page_number)
                elif newsCompany == self.irish_times_tag:
                    links = self.getITArticleLinks(topic, page_number)
                elif newsCompany == self.new_york_times_tag:
                    links = self.getNYTArticleLinks(topic, page_number)
        
                self.write_to_file(links, newsCompany, topic)
                print('ANOTHER PAGE OF LINKS EXTRACTED... ' + str(page_number) + ' FOR TOPIC ' + topic)

            print('TOPIC ' + topic + ' HAS FINISHED EXECUTING')

        # Finished writing articles, set to false
        self.pull_links = False
    
    def getIndependentArticleLinks(self, topic, page_number):
        links = []

        try:
            page_number = (page_number - 1) * 10

            url = ('https://www.independent.ie/search/?q={}&order=relevance&contextPublication=false&start={}'.format(topic, page_number))
                
            # Get the html for the page
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')

            # 1. Get the div with class c19
            articles_div = soup.find('div',attrs={'class':'n-split1-main'})
            
            # 2. Get all article divs (since each one contains a link)
            article_divs = articles_div.findAll('article',attrs={'class':'c-card1 -t:5'})



            # 3. Go through each article
            for article in article_divs:
                # 4. Get each URL
                for url in article.findAll('a'):
                    links.append(url.get('href'))

        except AttributeError:
            print('Failed to find what we looked for')
            print('Sleeping...')
            time.sleep(10)
        
        return links
    
    def getDailyMailArticleLinks(self, topic, page_number):
        links = []
        page_distance = 50
        off_set = until = 0

        # Unlike independent, daily mail does not store topic in news article and must append this manually
        url_apendix = "|" + topic

        try:
            off_set = (page_number - 1) * page_distance
            if page_number == 1:
                until = page_distance
            else:
                until = ((page_number - 1) * page_distance) * 2

            url = ('https://www.dailymail.co.uk/home/search.html?offset={}&size={}&sel=site&searchPhrase={}&sort=recent&type=article&type=video&type=permabox&days=all'.format(off_set, until, topic))

            # Get the html for the page
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')

            # 1. Get the articles div
            articles_div = soup.find('div',attrs={'class':'sch-results'})

            # Get all of the divs that contain links
            article_divs = articles_div.findAll('div')

            # 2. First 2 and last article div does not contain links
            for article in article_divs[2:-1]:
                for url in article.findAll('a'):
                    if '#' not in url.get('href'):
                        # Ensure the link has not already been added somehow
                        if not b_any(url.get('href') in link for link in links):
                            # print('https://www.dailymail.co.uk/' + url.get('href'))
                            links.append('https://www.dailymail.co.uk/' + url.get('href') + url_apendix)


        except AttributeError:
            print('Failed to find what we looked for')
            print('Sleeping...')
            time.sleep(10)

        return links  
    
    def getITArticleLinks(self, topic, page_number):

        return ''

    def getNYTArticleLinks(self, topic, page_number):
        return ''
    

    def getArticles(self, url, news_comapny):
        article = ''

        if news_comapny == self.independent_tag:
            article = self.getIndependentArticle(url)
        elif news_comapny == self.daily_mail_tag:
            article = self.getDailyMailArticle(url)
        elif news_comapny == self.irish_times_tag:
            article = self.getITArticle(url)
        elif news_comapny == self.new_york_times_tag:
            article = self.getNYTArticle(url)

        return article

    def getIndependentArticle(self, url, write=True):
        topic = ''
        article = ''
        # Header holds the date of article, url and author
        header = ''
        try:
            # Topic is appended to the end of URL, extract it for writing
            if '.html' in url:
                topic = url.split('.html')[1]
                # Take out the topic
                url = url[:-len(topic)]
            elif '.ece' in url:
                topic = url.split('.ece')[1]
                # Take out the topic
                url = url[:-len(topic)]
            else:
                return 'INVALID ARTICLE. Article could not be pulled'
            
            print('Pulling article for: ' + topic)

            response = requests.get(url) # , headers=ua)

            soup = BeautifulSoup(response.text, 'lxml')

            # Get the author
            author = soup.find('strong',attrs={'class':'name1'})

            # Get the date of the article
            date = soup.find('time',attrs={'class':'time1'})

            
            header = '{ ' + author.text + ' ' + date.text + ' ' + url + ' }' + '\n'
            

            # Get the paragraphs of the article
            div1 = soup.find('div',attrs={'class':'n-body1'})
            
            for para in div1.findAll('p'):
                article += ''.join(para.findAll(text=True)) + ' '
            with self.lock:
                # Write articles to corpus
                if write:
                    self.write_to_file(header + article, self.independent_tag, topic)
                else:
                    # Do not write as testing
                    pass

        except AttributeError:
            print('Failed to find what we looked for at link, ' + url)
            # When failed to read file, IP blocked,
            # send file to other dir and can read it later
            if write:
                self.write_dead_file_to_dir(url, self.independent_tag)
            else:
                # Skip as not writing
                pass

            print('Sleeping to stop IP block...')
            time.sleep(20)
        
        except UnicodeEncodeError:
            print('UniCodeEncodeError occured at link' + url)
            
        return article
    
    def makeConn(self, url):
        response = requests.get(url) # , headers=ua)
        soup = BeautifulSoup(response.text, 'lxml')

        title = soup.find('p',attrs={'div class':'news page-not-found scrollable-content cleared'})

        if title == "":
            return(soup)
        else:
            return("Could not make connection")

            

    def getDailyMailArticle(self, urlTopic, write=True):
        url, topic = urlTopic.split('|')
        article = ''
         # Header holds the date of article, url and author
        header = ''

        # Setup method
        if not write:
            self.call_tag = self.daily_mail_tag

        try:
            
            response = requests.get(url) # , headers=ua)
            

            soup = BeautifulSoup(response.text, 'lxml')

            # Get the author
            author = soup.find('p',attrs={'class':'author-section byline-plain'})

            # Get the date of the article
            date = soup.find('span',attrs='article-timestamp article-timestamp-published')
            
            header = '{ ' + author.text + ' ' + date.text + ' ' + url + ' }' + '\n'
            
            # Get the paragraphs of the article
            articleDivs = soup.find('div', attrs={'itemprop':'articleBody'})
            
            paras = articleDivs.findAll('p', attrs={'class':'mol-para-with-font'})
            
            for para in paras:
                article += ''.join(para.findAll(text=True)) + ' '
            
            if article == '':
                paras = articleDivs.findAll('p')
                for para in paras:
                    article += ''.join(para.findAll(text=True)) + ' '
            
            with self.lock:
                # Write articles to corpus
                if write:
                    self.write_to_file(header + article, self.daily_mail_tag, topic)
                else:
                    # Do not write for tests
                    pass

        except AttributeError:
            print('Failed to find what we looked for at link, ' + url)
            # When failed to read file, IP blocked,
            # send file to other dir and can read it later
            if write:
                self.write_dead_file_to_dir(url, self.daily_mail_tag)
            else:
                # Do not write for tests
                pass
            print('Sleeping to stop IP block...')
            # time.sleep(20)

        except UnicodeEncodeError:
            print('UniCodeEncodeError occured at link' + url)

        return article

    def getITArticle(self, utl):
        return ""

    def getNYTArticle(self, url):
        return ""

    
    def pull_articles_threader(self):
        while self.run:
            link = self.q.get()
            self.getArticles(link, self.call_tag)
            self.q.task_done()
    
    # Using threads pull articles
    def pull_articles(self, topic, tag=None):
        NUM_THREADS = 2
        self.call_tag = tag

        # Get all links in an object
        if self.call_tag == self.independent_tag:
            links = self.getCorpusLinks(self.independent_tag, topic)
        elif self.call_tag == self.daily_mail_tag:
            links = self.getCorpusLinks(self.daily_mail_tag, topic)
        elif self.call_tag == self.irish_times_tag:
            links = self.getCorpusLinks(self.irish_times_tag, topic)
        elif self.call_tag == self.new_york_times_tag:
            links = self.getCorpusLinks(self.new_york_times_tag, topic)

        for _ in range(NUM_THREADS):
            # Create the new thread using threader method
            t = threading.Thread(target = self.pull_articles_threader)
            t.daemon = True
            # Start the thread
            t.start()
        
        for link in links:
            self.q.put(link)
        
        self.q.join()
        self.run = False
        
    def getCorpusLinks(self, tag, topic):
        links = []
        DATA_PATH = "../newspaper_data/links/" + topic + "/"
        

        if tag == self.irish_times_tag:
            DATA_PATH = DATA_PATH + self.irish_times_tag + self.EXTENSION
        elif tag == self.independent_tag:
            DATA_PATH = DATA_PATH + self.independent_tag + self.EXTENSION
        elif tag == self.daily_mail_tag:
            DATA_PATH = DATA_PATH + self.daily_mail_tag + self.EXTENSION
        elif tag == self.new_york_times_tag:
            DATA_PATH = DATA_PATH + self.new_york_times_tag + self.EXTENSION

        newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_PATH))

        with open(newspaper_list_file, 'r') as filehandle:
            lines = filehandle.read().splitlines()
            
            if tag == self.daily_mail_tag:
                for line in lines:
                    links.append(line)
            else:
                for line in lines:
                    links.append(line + topic)
        
        return links
    
    def buildArticles(self, company_tag):

        for topic in self.refined_news_categories:
            if '.' in topic:
                topic = ' '.join(topic.split('.'))

            self.pull_articles(topic, company_tag)
            self.q = Queue()
            self.run = True
    
    # When IP blocked, the article is skipped and can read it from here in next iteration
    def write_dead_file_to_dir(self, url, tag):
        DATA_PATH = "../newspaper_data/backup_links/"

        if tag == self.independent_tag:
            DATA_PATH = DATA_PATH + self.independent_tag + self.EXTENSION
        elif tag == self.daily_mail_tag:
            DATA_PATH = DATA_PATH + self.daily_mail_tag + self.EXTENSION
        elif tag == self.irish_times_tag:
            DATA_PATH = DATA_PATH + self.irish_times_tag + self.EXTENSION
        elif tag == self.new_york_times_tag:
            DATA_PATH = DATA_PATH + self.new_york_times_tag + self.EXTENSION

        newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_PATH))

        with open(newspaper_list_file, 'a+') as filehandle:
            filehandle.write('%s\n' % url)

        
    # self.write_to_file(article, tag, topic)
    def write_to_file(self, news_data, tag=None, topic=""):
        DATA_PATH = ""
        # Evalues to fasle if string is empty
        if topic:
            topic = topic + '/'

        # Get the dirs for writing to a file
        if self.pull_links == True:
            DATA_PATH = "../newspaper_data/links/"
            FILE_PATH = ""

            if tag == self.irish_times_tag:
                DATA_PATH = DATA_PATH + topic
                FILE_PATH = '/' + self.irish_times_tag + self.EXTENSION
            elif tag == self.new_york_times_tag:
                DATA_PATH = DATA_PATH + topic
                FILE_PATH = '/' + self.new_york_times_tag + self.EXTENSION
            elif tag == self.independent_tag:
                DATA_PATH = DATA_PATH + topic
                FILE_PATH = '/' + self.independent_tag + self.EXTENSION
            elif tag == self.daily_mail_tag:
                DATA_PATH = DATA_PATH + topic
                FILE_PATH = '/' + self.daily_mail_tag + self.EXTENSION
        # Write the newspaper data (articles)
        else:
            DATA_PATH = "../corpus/irishArticles/"

            if tag == self.irish_times_tag:
                DATA_PATH = DATA_PATH + self.irish_times_tag + '/' + topic
            elif tag == self.new_york_times_tag:
                DATA_PATH =  DATA_PATH + self.new_york_times_tag + '/' + topic
            elif tag == self.independent_tag:
                DATA_PATH =  DATA_PATH + self.independent_tag + '/' + topic
            elif tag == self.daily_mail_tag:
                DATA_PATH =  DATA_PATH + self.daily_mail_tag + '/' + topic
            else:
                print('No article path specified!')

        newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_PATH))

        try:
            # Recursively build the dirs if they do not already exist
            os.makedirs(newspaper_list_file, exist_ok=True)
        except OSError:
            print('FAILED TO CREATE DIR RECURSIVELY')

        if self.pull_links == True: 
            # Write only the links, not array data
            with open(newspaper_list_file + FILE_PATH, 'a+') as filehandle:
                for link in news_data:
                    filehandle.write('%s\n' % link)
        else:
            # Get the number of articles (used to count the filenames)
            article_counter = self.getArticleCount(newspaper_list_file)
            with open(newspaper_list_file + '/' + str(article_counter), 'w') as filehandle:
                filehandle.write('%s\n' % news_data)
                article_counter = article_counter + 1

        print('FINISHED WRITING ARTICLES')
    
    def getArticleCount(self, newspaper_list_file):
        articles = os.listdir(newspaper_list_file)

        return(len(articles))



def main():
    articles = WebScrapeArticles()

    # Get links the independent
    articles.buildArticleLinks(articles.independent_tag)
    # Get articles for the independent
    articles.buildArticles(articles.independent_tag)

    # # Get links the Daily Mail
    # articles.buildArticleLinks(articles.daily_mail_tag)
    # Get articles for the daily mail
    # articles.buildArticles(articles.daily_mail_tag)

if __name__ == '__main__':
    main()

