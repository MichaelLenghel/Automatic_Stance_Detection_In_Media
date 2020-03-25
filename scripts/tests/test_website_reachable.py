import unittest
import sys, os
import subprocess


class test_generate_lda_model(unittest.TestCase):
    def setUp(self):
        sys.path.append('..')
        import scrape_articles as sa
        self.articles = sa.WebScrapeArticles()
    
    # Check Independent is reachable
    def test_independent_connection(self):
        self.assertNotEqual(self.articles.makeConn("https://www.independent.ie/"), "Could not make connection 404")
        self.assertNotEqual(self.articles.makeConn("https://www.independent.ie/"), "")
        
    # Check Daily Mail is reachable
    def test_daily_mail_connection(self):
        self.assertNotEqual(self.articles.makeConn("https://www.dailymail.co.uk/home/index.html"), "Could not make connection 404")
        self.assertNotEqual(self.articles.makeConn("https://www.dailymail.co.uk/home/index.html"), "")
    
    def test_python_version(self):
        python_version_pattern = "3.7"
        self.assertRegex(sys.version, python_version_pattern)

    def test_java_installed(self):
        # Java message that will be expected (Checking for version prone to fault)
        java_version_stub = "Java(TM) SE Runtime Environment"
        self.assertRegex(subprocess.run(["java", "-version"], stdout=subprocess.PIPE).stdout.decode('utf-8'), java_version_stub)


if __name__ == '__main__':
    unittest.main()