Setup Instructions (Fuctionality - Scraping aritcles, creating and running models):
1. Ensure python 3.7 is installed on the system (Does not work with any other minor version)
2. Ensure java 1.8 or later is installed and added to path (System uses mallet wrapper that has java as a dependency)
3. Inside current_python_deps folder run "pip install -r requirements.txt"
4. To scrape more articles can run scrape_articles.py. (Takes a long time.) Currently scraping articles for the independent and the daily mail are supproted (Optional Step)
5. Run generate_lda_model.py to create the associated word corpus (used for opinion mining tchniques) and lda topic corpus
6. Run visualise_lda.py to generate stance score. Can uncomment code responsible for calculating stance scores or build more visualisations if want more informaiton.

(Setup Instructions (Visualisations - Creating graphs with data)
1. Download mysql server from: https://dev.mysql.com/downloads/installer/
2. Start mysql service if not already started
3. Go to mysql server and start it with the command: "mysql.exe -u $user -p"
4. CREATE USER 'grafana'@'localhost' IDENTIFIED BY 'password';
5. CREATE DATABASE topic;
6. use topic;
7. GRANT SELECT ON topic.* TO 'grafana'@'localhost';
8. SOURCE DIR\database\sql_code\sentiments_tables.sql
9.Download grafana (Default port 3000)
10. open and add mysql as data source
11. Explore data and run "SELECT * FROM `topic`.`sentiments`;" with format as set to table to check data is reaching grafana
12. Run: "SELECT topic AS metric, avg_topic_word_sentiment as value FROM CREATE TABLE `topic`.`sentiments`(
ORDER BY id"