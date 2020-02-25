# This script will append the topic name of each news file within the folder
import os


def main():
    REAL_INDEPENDENT_PATH = '../corpus/irishArticles/INDEPENDENT'
    REAL_BBC_PATH = '../corpus/irishArticles/BBC'
    

    INDEPENDENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), REAL_INDEPENDENT_PATH))
    
    # Get topics so we can itertae through them
    topics = os.listdir(INDEPENDENT_PATH)

    for topic in topics:
        print('REWRITING: ' + topic)

        TOPIC_PATH = INDEPENDENT_PATH + '\\' + topic
        print('asdsadasdasdsad' + TOPIC_PATH)
        

        # All the articles within a topic
        
        articles = os.listdir(TOPIC_PATH)

        for article in articles:
            print('REWRITING SPECIFICALLY: ' + article)
            with open(TOPIC_PATH + '/' + article, 'r+') as filehandle:
                original_file = filehandle.read()
                filehandle.seek(0) # rewind
                filehandle.write(topic + '\n' + original_file)


if __name__ == '__main__':
    main()