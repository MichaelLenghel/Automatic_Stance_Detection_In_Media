import os
import sys

def sort_topic_name(topic_name):
    return topic_name[1]

def format_sentiments(company_tag):
    SENTIMENTS_PATH = "../newspaper_data/sentiments/" + company_tag + '.txt'
    FULL_SENTIMENTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), SENTIMENTS_PATH))

    topics = []

    with open(FULL_SENTIMENTS_PATH, 'r') as filehandle:
        for line in filehandle:
            line_parts = line.split(' ')
            topic_name = line_parts[3]
            topic_word_sent = line_parts[8][:6]
            avg_word_sentiment = line_parts[11][:6]
            avg_topic_objectivity = line_parts[25][:6]

            topic_info = ("Topic: " + topic_name + " Topic Word Sentiment: " + topic_word_sent + " Average Word Sentiment: " + avg_word_sentiment + " Average Word Objectivity " + avg_topic_objectivity + "\n", topic_name)
            topics.append(topic_info)
    
    topics.sort(key = sort_topic_name)

    for topic in topics:
        print(topic[0])


def main():

    companies = ["INDEPENDENT", "DAILY_MAIL"]

    if len(sys.argv) < 2:
        print("Must pass a company as argument. Accepted companies: [ " + companies[0] + ", " + companies[1] + " ]")
        return ''
    else: 
        company = sys.argv[1]

        if company in companies:
            format_sentiments(company)
        else:
            print("Must pass a company as argument. Accepted companies: [ " + companies[0] + ", " + companies[1] + " ]")    



if __name__ == '__main__':
    main()