import unittest
import sys

class test_scrape_articles(unittest.TestCase):
    def setUp(self):
        sys.path.append('..')
        import scrape_articles as sa
        self.articles = sa.WebScrapeArticles()
        self.reg_pattern = "http"
        self.topic = 'brexit'
        self.page_number = 1
        self.url_independent = 'https://www.independent.ie/world-news/coronavirus/met-eireann-predicts-sunny-dry-week-but-some-closures-announced-as-outdoor-revellers-ignore-social-distancing-advice-39065233.html|Brexit'
        self.url_daily_mail = 'https://www.dailymail.co.uk/tvshowbiz/article-8141933/Sneaky-Pete-star-Giovanni-Ribisi-covers-face-mask-gloves-stocks-supplies.html?ns_mchannel=rss&ico=taboola_feed|Brexit'
    
    def test_getDailyMailArticleLinksNotEmpty(self):
        # Check returns an element
        self.assertNotEqual(self.articles.getDailyMailArticleLinks(self.topic, self.page_number), [])
    
    def test_getDailyMailArticleLinksCorrectData(self):
        # Check includes correct syntax
        self.assertRegex("". join(self.articles.getDailyMailArticleLinks(self.topic, self.page_number)), self.reg_pattern)
    
    def test_getIndependentArticleLinksNotEmpty(self):
        # Check returns an element
        self.assertNotEqual(self.articles.getIndependentArticleLinks(self.topic, self.page_number), [])
    
    def test_getIndependentArticleLinksCorrectData(self):
        # Check includes correct syntax
        self.assertRegex("". join(self.articles.getIndependentArticleLinks(self.topic, self.page_number)), self.reg_pattern)
    
    def test_getIndependentArticleNotEmpty(self):
        # Check returns a non empty string
        self.assertNotEqual(self.articles.getIndependentArticle(self.url_independent, False), '')
    
    def test_getIndependentArticleCorrectData(self):
        # Check includes a sentence from the article
        sentence_pattern = "Spring has sprung, but this year, the hope and enthusiasm that usually comes with it has been subdued by the coronvirus crisis and the unknown prospect of what that might bring."
        self.assertRegex(self.articles.getIndependentArticle(self.url_independent, False), sentence_pattern)
    
    def test_getDailyMailArticleNotEmpty(self):
        # Check returns a non empty string
        self.assertNotEqual(self.articles.getDailyMailArticle(self.url_daily_mail, False), '')
    
    def test_getDailyMailArticleCorrectData(self):
        # Check includes a sentence from the article
        sentence_pattern = "Giovanni Ribisi made his health his priority."
        self.assertRegex(self.articles.getDailyMailArticle(self.url_daily_mail, False), sentence_pattern)



if __name__ == '__main__':
    unittest.main()