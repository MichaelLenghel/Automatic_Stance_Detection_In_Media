Setup Instructions:
1. Ensure python 3.7 is installed on the system (Does not work with any other minor version)
2. Ensure java 1.8 or later is installed and added to path (System uses mallet wrapper that has java as a dependency)
3. Inside python_deps folder run virtualenv ./Script/activate in windows or in linux source ./Script/activate (Sets packages) 
4. To scrape more articles can run scrape_articles.py. (Take a long time.) Currently scraping articles for the independent and the daily mail are supproted (Optional Step)
5. Run generate_lda_model.py to create the associated word corpus (used for opinion mining tchniques) and lda topic corpus
6. Run visualise_lda.py to generate stance score. Can uncomment code responsible for calculating stance scores or build more visualisations if want more informaiton.
