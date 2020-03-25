# This script will append the topic name of each news file within the folder
import os
import sys
import getopt


def main(argv):
    INDEPENDENT_PATH = '../corpus/irishArticles/INDEPENDENT'
    DAILY_MAIL_PATH = '../corpus/irishArticles/DAILY_MAIL'
    
    PATH = ""

    try:
        opts, args = getopt.getopt(argv,"c:",["company_name"])
    except getopt.GetoptError:
        print("Exception occured. Pass in the name of a company to continue.\n\nAccepted companies are \"independent\" and \"daily_mail\"")
        sys.exit(2)
    
    if len(sys.argv) == 1:
        print("Must pass arguments")
        return
    
    for opt, arg in opts:
        if opt in ('-c' or '-company_name'):
            if arg == 'independent':
                PATH = INDEPENDENT_PATH
            elif arg == 'daily_mail':
                PATH = DAILY_MAIL_PATH
            else:
               print("Specify a company that exists")
               sys.exit(2) 

    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), PATH))
    
    # Get topics so we can itertae through them
    topics = os.listdir(PATH)

    for topic in topics:
        print('REWRITING: ' + topic)

        TOPIC_PATH = PATH + '\\' + topic
        
        # All the articles within a topic
        
        articles = os.listdir(TOPIC_PATH)

        for article in articles:
            print('REWRITING SPECIFICALLY: ' + article)
            with open(TOPIC_PATH + '/' + article, 'r+') as filehandle:
                original_file = filehandle.read()
                filehandle.seek(0) # rewind
                filehandle.write(topic + '\n' + original_file)


if __name__ == '__main__':
    main(sys.argv[1:])