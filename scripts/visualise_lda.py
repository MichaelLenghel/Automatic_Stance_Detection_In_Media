import os
import pickle
from pprint import pprint
from collections import defaultdict
import copy
import sys
import getopt

# NLP libraries
import gensim
import gensim.corpora as corpora
import pandas as pd
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess, lemmatize
from gensim.models import CoherenceModel, KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import re
from sklearn.datasets import load_files
from nltk.corpus import wordnet as wn
import textblob as tb
import generate_lda_model
from collections import Counter
from decimal import Decimal


# Visualisation libraries.
import pyLDAvis.gensim

# Get the model from genism path
def retrieve_modal(path = None):
    if path is None:
        model_file = datapath("model")
    else:
        model_file = datapath(path)
        
    # Load
    lda = gensim.models.ldamodel.LdaModel.load(model_file)

    print('Finished retrieving model...')
    return lda

def retrieve_word_article_list(company_tag=None):
    extension = '.txt'

    if company_tag is None:
        MODEL_PATH = "../newspaper_data/newspaper_list.txt"
    else:
        MODEL_PATH = "../newspaper_data/model_data/" + company_tag + extension

    newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))

    # MODEL_PATH_WRITE = "../newspaper_data/newspaper_list_writing.txt"
    # newspaper_list_file_write = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH_WRITE))

    newspaper_article_list = []
    newspaper_word_list = []

    with open(newspaper_list_file, 'r') as filehandle:
        for line in filehandle:
            # String splitting removes first bracket and new line + closing bracket
            current_line = line[1:-2]

            # Split each word into the list
            current_list = current_line.split(', ')

            for word in current_list:
                # Append the word and remove closing and opening quotation
                newspaper_word_list.append(word[1:-1])
                
            newspaper_article_list.append(newspaper_word_list)
            newspaper_word_list = []

    print('Finished retrieving article word list...')

    # newspaper_article_list.pop()
    return newspaper_article_list

def build_dictionary_corpus(article_word_list):
    # Make word_dict & corpus
    word_dict = corpora.Dictionary(article_word_list)
    corpus = [word_dict.doc2bow(data) for data in article_word_list]

    print('Finished creating word_dict and corpus...')
    return word_dict, corpus

def build_lda_model(articles, word_dict, corpus):
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=word_dict,
                                            num_topics=50, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
    
    return lda_model

def compute_complexity(article_word_list, word_dict, corpus, lda):
    # Coherence score is the probability that a word has occured in a certain topic, so the quality of the topic matching.
    coherence_model_lda = CoherenceModel(model=lda, texts=article_word_list, dictionary=word_dict, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score:', coherence_lda)

    

    # Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus.
    # Also output the calculated statistics, including the perplexity=2^(-bound), to log at INFO level.
    # corpus_csc = gensim.matutils.corpus2csc(corpus, num_terms=len(fnames_argsort))
    # corpus_csc = gensim.matutils.corpus2csc(corpus, num_terms=19)
    # perplexity = lda.log_perplexity(corpus_csc)
    # print('Perplexity:', perplexity)  # a measure of how good the model is. lower the better.
    

def build_lda_visualizaiton(corpus, word_dict, lda):
    MODEL_PATH = "../topic_visualisaiton/visualisation.html"
    topic_visualisation = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))

    # Extracts info from lda model for visualisaition
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, word_dict, sort_topics=False)
    print('Finished Prepare!!')

    pyLDAvis.show(lda_display)
    ('Finished display!')
    # Save graph so it can be used. IPYTHON dependancy required to display
    pyLDAvis.save_html(lda_display, topic_visualisation)
    print('Finished saving visualization...')

# Found this method online, lost reference, but used some of the logic
def format_topics_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()

    print('Formatting Document...')

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        ## When using mallet:
        # Sort the rows, as the ones at position zero will be the highest value
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        ## When using regular lda (not mallet):
        # row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => The dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])

                # print('Topic num = ' + str(topic_num) + ' Keywords = ' + topic_keywords )
                
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def get_raw_data(news_path, company_tag):
    refined_news_categories = ['brexit', 'climate.change', 'financial'
                , 'trump', 'sport.gaa' ,'sport.football'
                , 'food.reviews', 'polotics', 'medicine'
                , 'middle.east', 'abortion'
                , 'atheism', 'christianity', 'drugs'
                , 'USA', 'china', 'business'
                , 'housing', 'online']

    # Setup path to test corpus
    NEWS_GROUPS_TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), news_path))

    news = load_files(NEWS_GROUPS_TEST_PATH, description='Newspaper Topics from the specified company'
                                , categories=refined_news_categories, load_content=True , shuffle=True, encoding='latin1'
                                , decode_error='strict', random_state=30)
    



    news = list(news.data)

    # Remove Emails
    news = [re.sub('\S*@\S*\s?', '', sent) for sent in news]

    # Remove new line characters
    news = [re.sub('\s+', ' ', sent) for sent in news]

    # Remove distracting single quotes
    news = [re.sub("\'", "", sent) for sent in news]

    return news

def build_mallet_lda_model(articles, word_dict, corpus):
    from gensim.models.wrappers import LdaMallet

    MALLET_PATH = '../mallet_2/mallet-2.0.8/bin/mallet.bat'
    mallet_file = os.path.abspath(os.path.join(os.path.dirname(__file__), MALLET_PATH))
    os.environ.update({'MALLET_HOME':mallet_file})

    lda_mallet_model = gensim.models.wrappers.LdaMallet(mallet_file, corpus=corpus, num_topics=19, id2word=word_dict)

    print('CREATED MALLET MODEL')
    return lda_mallet_model

def print_topic_words(lda_mallet_model):
    # # Display topics and simply list words without weights for more clarity
    for index, topic in lda_mallet_model.show_topics(num_topics=19, formatted=False): #, num_words= 30):
        topic_word = ''

        # Hypernamy technique:
        # # Grab the top words from the list
        for i, w in enumerate(topic, 4):
            if i == 0:
                topic_word = w[0]
                break

        # print('TOPWORD: ' + top_word)
        # nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}


        
        # if top_word in nouns and second_word in nouns:
        #     # Get the hypernym for the words with the two largest weights
        #     topic_word = wn.synset(top_word + '.n.01').lowest_common_hypernyms(wn.synset(second_word + '.n.01'))
        # else:
        #     topic_word = topic_word + ' | ' + second_word

        print('Most likely topic: {} {} \nMost weighted Words: {}'.format(index, topic_word, '|'.join([w[0] for w in topic])))

def getSentiment(x):
    # print(x)
    t = tb.TextBlob(x)
    return t.sentiment.polarity, t.sentiment.subjectivity

def most_frequent_topic(list):
    return max(set(list), key = list.count)

def topic_fight(imaginery_topic, topic, topic_word_dict, word_corpus):
    winner = ""
    losing_topic = ""
    imaginery_topic_list = []
    existing_topic_list = []

    # Get number of topic occurences of imaginery topic
    for i , w in enumerate(imaginery_topic):
        # Do for only first 10 words, but try with 30 and check if accuracy is changing
        if i == 10:
            break
        
        if w[0] in word_corpus.keys():
            for word in word_corpus[w[0]]:
                # Grab the topic where the word is found
                word_parts = word.split(':::')
                imaginery_topic_list.append(word_parts[0])
            imaginery_topic_count = imaginery_topic_list.count(topic)
    
    # Get number of topic occurences of existing topic
    for i , w in enumerate(topic_word_dict[topic]):
        # Do for only first 10 words, but try with 30 and check if accuracy is changing
        if i == 10:
            break
        
        if w[0] in word_corpus.keys():
            for word in word_corpus[w[0]]:
                # Grab the topic where the word is found
                word_parts = word.split(':::')
                existing_topic_list.append(word_parts[0])
            existing_topic_count = existing_topic_list.count(topic)
    
    if imaginery_topic_count > existing_topic_count:
        winner = "IMAGINERY"
        existing_topic_list.remove(topic)
        losing_topic = most_frequent_topic(existing_topic_list)
    else:
        winner = "EXISTING"
        imaginery_topic_list.remove(topic)
        losing_topic = most_frequent_topic(imaginery_topic_list)
    
    return winner, losing_topic


# Will return the topic that most matches the given 30 words
def getRealTopic(imaginery_topic, word_corpus, used_topic_words, topic_word_dict, fight):
    real_topic = ''
    topic_list = []
    winner = ""
    losing_topic = ""

    for i , w in enumerate(imaginery_topic):
        # Do for only first 10 words, but try with 30 and check if accuracy is changing
        if i == 10:
            break
        print("Before adding real topic: ")
        # For w[0] get all sentences where the word is used (from a premade corpus dictionary that matches a word to each occurence and the topic name in the folder)
        if w[0] in word_corpus.keys():
            for word in word_corpus[w[0]]:
                # Grab the topic where the word is found
                word_parts = word.split(':::')
                topic_list.append(word_parts[0])
            real_topic = most_frequent_topic(topic_list)
            print("Real topic is: " + real_topic)

            if fight:
                # If this topic already, fight them to find stronger one
                if real_topic in used_topic_words:
                    winner, losing_topic = topic_fight(imaginery_topic, real_topic, topic_word_dict, word_corpus)
                else:
                    return real_topic, "", ""

        # If winner == "EXISTING" -> no need to replace topic_word_dict
        # If winner == "IMAGINERY" -> need to replace in topic_word_dict
        return real_topic, winner, losing_topic

# Topic and word that equates to topic and sentence in array
def get_topic_sentences(topic, topic_sentences):
    topical_sentences = []

    for topic_sentence in topic_sentences:
        topic_sentence_parts = topic_sentence.split(':::')
        sentence_topic = topic_sentence_parts[0]
        full_sentence = topic_sentence_parts[3]

        if sentence_topic == topic:
            topical_sentences.append(full_sentence)
        else:
            continue
    
    return topical_sentences

def defaultTopicValue():
    return []

def calculate_sentiment_mode(polarity, subjectivity, avg_polarity, avg_sentiment):
    # Calculate the mode to a one number degree (round)
    polarity_data = Counter(polarity)
    subjectivity_data = Counter(subjectivity)

    get_mode_polarity = dict(polarity_data)
    get_mode_subjectivity = dict(subjectivity_data)

    polarity_mode = [k for k, v in get_mode_polarity.items() if v == max(list(get_mode_polarity.values()))]
    subjectivity_mode = [k for k, v in get_mode_subjectivity.items() if v == max(list(get_mode_subjectivity.values()))]

    # Account for multiple lists
    if type(polarity_mode) is list:
        polarity_mode = polarity_mode[0]
    
    if type(subjectivity_mode) is list:
        subjectivity_mode = subjectivity_mode[0]


    return polarity_mode, subjectivity_mode

def calculate_sentiment_median(polarity, subjectivity):
    # Calculate median
    polarity_sorted = sorted(polarity)
    polarity_median = polarity_sorted[int(len(polarity) / 2)]
    subjectivity_sorted = sorted(subjectivity)
    subjectivity_median = subjectivity_sorted[int(len(subjectivity) / 2)]

    return polarity_median, subjectivity_median

# def get_topic_sentences2(topic_name, ldamodel):
#       for i, row in enumerate(ldamodel[corpus]):
#         row = sorted(row, key=lambda x: (x[1]), reverse=True)

# Uses top 30 words per topic to calculate polarity and subjectivity of each topic
def correlate_top_words_sentiment(lda_mallet_model, word_corpus, company_tag, fight=False):
    topic_sentiments = {}
    topic_word_dict = defaultdict(defaultTopicValue)
    topic_name_polarity = []
    topic_name_subjectivity = []
    word_list = []
    polarity = []
    subjectivity = []
    isSciObj = False
    used_topic_words = []
    counter = 1

    # Try for 10 num words as well as 30 and see which is more  accurate
    for index, topic in lda_mallet_model.show_topics(num_topics=19, formatted=False, num_words=15): #, num_words= 15):
        imaginery_topic = topic

        # getRealTopic. Can remove code involing fightingTopic and will make it more accurate as only a few topics will come through
        # But this increases how many results come through. Delete any duplicate topics that mismatch necessary using fightingTopics
        topic_name, winner, losing_topic = getRealTopic(imaginery_topic, word_corpus, used_topic_words, topic_word_dict, fight)

        if fight:
            # The duplicate has been fixed by getRealTopic to another topic, replace topic_name to losing_topic and perform operations for it
            if winner == "EXISTING":
                topic_name = losing_topic
            elif winner == "":
                # If winner and losing_topic are empty, that means this is a new topic and dont need to replce variables
                pass
            else:
                # The imaginery topic is a stronger topic than the existing one
                # Firstly move the existing topic which is weaker then current one to new topic and append a number as it may also be beaten by another topic
                topic_sentiments[losing_topic + "L" + str(counter)] = copy.deepcopy(topic_sentiments[topic_name])
                topic_word_dict[losing_topic +  "L" + str(counter)] =  copy.deepcopy(topic_word_dict[topic_name])
                counter = counter + 1

        used_topic_words.append(topic_name)
        print('Topic name is: ' + topic_name)

        for w in topic:
            if w[0] in word_corpus.keys():

                ### Algo to remove generic words goes here

                # Only get the w[0] of the topic that has topic_name in the sentence of that word
                sentences = get_topic_sentences(topic_name, word_corpus[w[0]])
                # sentences = get_topic_sentences2(topic_name, lda_mallet_model)
                # Add the word to used words (these are words we are checking sentiment for)
                word_list.append(w[0])

                sentences_cleaned_list = generate_lda_model.clean_data(sentences, company_tag, isSciObj)

                for sentence in sentences_cleaned_list:
                    sentence_str = ' '.join([str(word) for word in sentence])
                    pol, subjec = getSentiment(sentence_str)
                    # many zeros from falsely understood sentences
                    # print("Adding polarity: " + str(pol))
                    # print("Adding subjectivity: " + str(subjec))

                    # Do not include zeros as they skew data too close to zero
                    # and does not show the weight of bias when used.
                    # This allows to discern opinion more easily
                    if pol != 0:
                        polarity.append(pol)
                        if topic_name in sentence_str:
                            topic_name_polarity.append(pol)
                    if  subjec != 0:
                        subjectivity.append(subjec)
                        if topic_name in sentence_str:
                            topic_name_subjectivity.append(subjec)
            else:
                continue
            
        # Calculate average, mode and median for polarity and sentiment
        if len(polarity) != 0 and len(subjectivity) != 0:
            avg_polarity = sum(polarity) / len(polarity)
            avg_subjectivity = sum(subjectivity) / len(subjectivity)
        if len(topic_name_polarity) != 0 and len(topic_name_subjectivity) != 0:
            avg_topic_name_polarity = sum(topic_name_polarity) / len(topic_name_polarity)
            avg_topic_name_subjectivity = sum(topic_name_subjectivity) / len(topic_name_subjectivity)
        else:
            avg_topic_name_polarity = None
            avg_topic_name_subjectivity = None

        polarity_mode, subjectivity_mode = calculate_sentiment_mode(polarity, subjectivity, avg_polarity, avg_subjectivity)
        polarity_median, subjectivity_median = calculate_sentiment_median(polarity, subjectivity)
        # Add values to sentiment dictionary, as well as words
        topic_sentiments[topic_name + str(counter)] = [avg_topic_name_polarity, avg_polarity, polarity_mode, polarity_median, avg_topic_name_subjectivity, avg_subjectivity, subjectivity_mode, subjectivity_median]
        topic_word_dict[topic_name + str(counter)].append(word_list[:])
        counter = counter + 1

        print('Added topic: ' + topic_name)
        print('For topic {}, probably: {} the average polarity is {} and the average subjectivity is {}'.format(index, topic_name, avg_polarity, avg_subjectivity))

        # Clean up
        word_list[:] = []
        polarity[:] = []
        subjectivity[:] = []
        topic_name_polarity[:] = []
        topic_name_subjectivity[:] = []

    
    return (topic_sentiments, topic_word_dict)

def retrieve_word_corpus(company_tag):
    WORD_CORPUS_PATH = "../newspaper_data/word_corpus/" + company_tag + '.txt'

    FULL_WORD_CORPUS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), WORD_CORPUS_PATH))

    with open(FULL_WORD_CORPUS_PATH, 'rb') as myFile:
         word_corpus = pickle.load(myFile)
    
    return word_corpus

def write_topic_sentiments(topic_sentiments, topic_word_dict, company_tag):
    SENTIMENTS_PATH = "../newspaper_data/sentiments/" + company_tag + '.txt'

    FULL_SENTIMENTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), SENTIMENTS_PATH))

    word_li = []

    with open(FULL_SENTIMENTS_PATH, 'w') as myFile:
        for index, topic_name in enumerate(topic_sentiments):
            for words in topic_word_dict:
                for word in topic_word_dict[words]:
                    word_li.append(word)
            topic_info = "Index: {} Topic: {} Topic Word Average Sentiment: {} Average Sentiment: {} Mode Sentiment: {} Median Sentiment: {} Topic Word Average Subjectivity: {} Average Subjectivity: {} Mode Subjecivity: {} Median Subjectivity: {} Words: {}\n".format(
                index, topic_name, topic_sentiments[topic_name][0], topic_sentiments[topic_name][1], topic_sentiments[topic_name][2], topic_sentiments[topic_name][3], topic_sentiments[topic_name][4],
                topic_sentiments[topic_name][5], topic_sentiments[topic_name][6], topic_sentiments[topic_name][7], topic_word_dict[topic_name])

            myFile.write(topic_info)


def main(argv):
    FIND_TAG = {'#SAD23rgA'}
    company_tags = ['INDEPENDENT', 'DAILY_MAIL', 'NEW-YORK-TIMES']
    mallet_tag = 'mallet'
    REAL_INDEPENDENT_PATH = '../corpus/irishArticles/INDEPENDENT'

    company = ""

    try:
        opts, args = getopt.getopt(argv,"c:",["company_name"])
    except getopt.GetoptError:
        print("Exception occured. Pass in the name of a company to continue.\n\nAccepted companies are \"independent\" and \"daily_mail\"")
        sys.exit(2)
    
    if len(sys.argv) == 1:
        print("Must pass arguments -c company_name.\n\nAccepted companies are \"independent\" and \"daily_mail\"")
        return
    
    for opt, arg in opts:
        if opt in ('-c' or '-company_name'):
            if arg == 'independent':
                company = company_tags[0]
            elif arg == 'daily_mail':
                company = company_tags[1]
            else:
               print("Specify a company that exists")
               sys.exit(2)
    
    print("Creating model for " + company)

    # Recover list data
    article_word_list = retrieve_word_article_list(company)
    
    # Create dictionary, corpus
    word_dict, corpus = build_dictionary_corpus(article_word_list)

    # # Retrieve ldamodel
    # lda = retrieve_modal(company)

    # Retrieve lda_mallet_model. Save and retrieval does not work properly for mallet model due to some issues, so must remake mallet model.
    lda_mallet_model = build_mallet_lda_model(article_word_list, word_dict, corpus)

    word_corpus = retrieve_word_corpus(company)

    # for word in word_corpus:
    #     print('WORD: ' + word + ' POINTS TO ', word_corpus[word])
    
    # At the moment not using full sentences, need method to retrieve full sentence
    # Use textblob and correlate top used words to their sentiment and average per topic
    topic_sentiments, topic_word_dict = correlate_top_words_sentiment(lda_mallet_model, word_corpus, company)

    write_topic_sentiments(topic_sentiments, topic_word_dict, company)

    # print_topic_words(lda_mallet_model)

    # # Find the dominant topic
    # df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_mallet_model, corpus=corpus, texts=article_word_list[:-1])

    # df_dominant_topic = df_topic_sents_keywords.reset_index()
    # df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    # print(df_dominant_topic.head(10))
    # # # Find the most representive document for each topic
    # # Group top 5 sentences under each topic ###################
    # sent_topics_sorteddf_mallet = pd.DataFrame()

    # sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    # for i, grp in sent_topics_outdf_grpd:
    #     sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
    #                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
    #                                             axis=0)

    # # Reset Index    
    # sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # # Number of Documents for Each Topic
    # topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

    # # Percentage of Documents for Each Topic
    # topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # # Topic Number and Keywords
    # topic_num_keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]

    # # Concatenate Column wise
    # df_dominant_topics = pd.concat([topic_num_keywords, topic_counts.sort_index(), topic_contribution.sort_index()], axis=1)

    # # Change Column names
    # df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    # print(df_dominant_topics)



    
    ####### Compute complexity (Left out as long execution) ########
    # compute_complexity(article_word_list, word_dict, corpus, lda)

    # Below method only works in a shell such as python shell. Also takes about 15 minutes
    # import visualise_lda : Then run each command in console separetly with visualise_lda prefix
    # for mallet diff (https://stackoverflow.com/questions/50340657/pyldavis-with-mallet-lda-implementation-ldamallet-object-has-no-attribute-inf), which says convert it to lda model:
    # import gensim    
    # model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mallet_model)
    # build_lda_visualizaiton(corpus, word_dict, lda_mallet_model)

    # get_document_topics with a single token 'religion'
    # text = ["trump"]
    # bow = word_dict.doc2bow(text)
    # model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_mallet_model)
    # pprint(model.get_document_topics(bow))

    # # Show top 30 topics (with weights)
    # pprint(FIND_TAG + lda_mallet_model.show_topics(30))

    ## Does not work Below, only test!!!!
    # # Format
    # sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    # vals = sent_topics_sorteddf_mallet.Keywords.progress_apply(sent)

    # sent_topics_sorteddf_mallet['polarity'] = vals.str[0]
    # sent_topics_sorteddf_mallet['sub'] = vals.str[1]


if __name__ == '__main__':
    main(sys.argv[1:])

