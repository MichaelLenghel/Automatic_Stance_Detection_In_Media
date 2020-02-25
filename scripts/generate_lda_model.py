import os
import re
import pickle
from pprint import pprint

# NLP libraries
import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel, KeyedVectors
import spacy
import pattern
import pandas as pd
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# Include extra words. May in the future extend it even further
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# sklearn for accessing 20 news groups
from sklearn.datasets import load_files

def gen_bunch(news_path, company_tag):
    # Cateogies in data set
    news_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc'
                , 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'
                , 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball'
                , 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med'
                , 'sci.space', 'soc.religion.christian', 'talk.politics.guns'
                , 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    
    refined_news_categories = ['brexit', 'climate.change', 'financial'
                , 'trump', 'sport.gaa' ,'sport.football'
                , 'food.reviews', 'polotics', 'medicine'
                , 'middle.east', 'abortion'
                , 'atheism', 'christianity', 'drugs'
                , 'USA', 'china', 'business'
                , 'housing', 'online']

    # Setup path to test corpus
    NEWS_GROUPS_TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), news_path))

    # Print the path
    # print(NEWS_GROUPS_TEST_PATH)


    ##### Need to implement a method for including custom categories! ####

    # Load all test data.

    # news_test = load_files(NEWS_GROUPS_TEST_PATH, description='News Paper Test Topics from 20 news groups'
    #                             , categories=news_categories, load_content=True , shuffle=False, encoding='latin1'
    #                             , decode_error='strict')

    # Shuffling the data in order to increase distribution of topics and not overly simplify NLP patterns
    # news_test.data is everything in one big string

    if company_tag == 'TEST':
        news = load_files(NEWS_GROUPS_TEST_PATH, description='Newspaper test Topics from 20 news groups'
                                    , categories=news_categories, load_content=True , shuffle=True, encoding='latin1'
                                    , decode_error='strict', random_state=30)
    else:
        news = load_files(NEWS_GROUPS_TEST_PATH, description='Newspaper Topics from the specified company'
                                    , categories=refined_news_categories, load_content=True , shuffle=True, encoding='latin1'
                                    , decode_error='strict', random_state=30)
    
    # Note:
    # Shows the topic and document ID + the article.
    # print(news.filenames[0])
    # print(news.data[0])

    # Get all of the file names
    # for integer_category in news_test.target[:10]:
    #     print(news_test.target_names[integer_category])

    return news

def multiple_replacements(article):
    empty_str = ""

    # Replacing all dashes, equals, cursors
    replacements = {
        "-" : empty_str,
        "=": empty_str,
        "^": empty_str,
    }

    # Replace newlines
    article_list = re.sub('\s+', ' ', article)

    # Replace emails
    article_list = re.sub('\S*@\S*\s?', '', article_list)

    # Replace quotes
    article_list = re.sub("\'", "", article_list)

    # Replace headers of data (author, date and url)
    article_list = re.sub(r'\{[^)]*\}', '', article_list)

    # Create a regular expression using replacements and join them togetehr.
    # re.compile creates the pattern object
    # re.escape avoids using special characters in regex
    reg = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))

    # For each match, look-up corresponding value in dictionary
    return reg.sub(lambda value: replacements[value.string[value.start():value.end()]], article)


def split_to_word(articles):
    # Iterate over every article 
    for article in articles:
        # Yield to not overload the memory with the big data set.
        # Deacc parameter removes all punctuations as well as spliting each word.
        yield(gensim.utils.simple_preprocess(str(article), deacc=True))

def create_bigrams(articles, bigram_model):
    return [bigram_model[article] for article in articles]

def remove_stopwords(articles):
    return [[w for w in simple_preprocess(str(article)) if w not in stop_words] for article in articles]

def lemmatize_words(bigram_model):
    # Only considers nouns, verbs, adjectives and adverbs
    return [[w for w in lemmatize(str(article))] for article in bigram_model]

# This method is about a minute faster for a data set of 7000 than the one above
def lemmatization(articles, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    articles_lem = []

    # Load the spacy lammatixation model for english 
    spacy_lem = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    for article in articles:
        w = spacy_lem(" ".join(article)) 
        articles_lem.append([token.lemma_ for token in w if token.pos_ in allowed_postags])

    return articles_lem

def build_lda_model(articles, word_dict, corpus):
    # Build LDA model
    # Retry with random_state = 0
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

# Build more accurate lda model using mallet
def build_mallet_lda_model(articles, word_dict, corpus):
    from gensim.models.wrappers import LdaMallet

    MALLET_PATH = '../mallet_2/mallet-2.0.8/bin/mallet.bat'
    mallet_file = os.path.abspath(os.path.join(os.path.dirname(__file__), MALLET_PATH))
    os.environ.update({'MALLET_HOME':mallet_file})

    lda_mallet_model = gensim.models.wrappers.LdaMallet(mallet_file, corpus=corpus, num_topics=19, id2word=word_dict)

    return lda_mallet_model

def clean_data(test_bunch, company_tag, isSciObj=True):
    ########## ALGO ###################

    # Create a list of the articles
    if isSciObj == False:
        article_list = list(test_bunch)
    else:
        article_list = list(test_bunch.data)
    
    # Replace =, cursor and dash, quotes, newlines and emails
    article_word_list = [multiple_replacements(article) for article in article_list]

    # split each article into words
    article_word_list = list(split_to_word(article_word_list))

    # Print the first article word by word:
    # print(article_word_list[0])

    # brigrams model
    # Need bigrams as it cuts word that go together down into one
    bigram = gensim.models.Phrases(article_word_list, min_count=8, threshold=100)

    bigram_model = gensim.models.phrases.Phraser(bigram)

    # Print bigrams
    # print(bigram_model[article_word_list[0]])

    # Remove stopwords
    article_no_stopwords = remove_stopwords(article_word_list)

    # make bigrams
    bigram_words = create_bigrams(article_no_stopwords, bigram_model)

    # Lemmatize - By default only nouns, verbs, adjectives and adverbs
    # lemmatized_article = lemmatize_words(bigram_words)

    lemmatized_article = lemmatization(bigram_words, allowed_postags=['NOUN', 'VERB', 'ADJ',  'ADV'])

    return lemmatized_article

def save_model(lda_model, path = None, model_type = None):
    # Below doesn't work due to access denied issues, datapath is the alternative

    # MODEL_PATH = "../models/"
    # model_file = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))

    if path is None:
        model_file = datapath("model")
    else:
        model_file = datapath(path)

    lda_model.save(model_file)

def save_data(newspaper_list, company_tag):
    MODEL_PATH = "../newspaper_data/newspaper_list.txt"
    COMPANY_MODEL_PATH = "../newspaper_data/model_data/"
    extension = ".txt"
    COMPANY_PATH = company_tag + extension

    if company_tag == 'TEST':
        newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))
    else:
        newspaper_list_file = os.path.abspath(os.path.join(os.path.dirname(__file__), COMPANY_MODEL_PATH))
    
    try:
        # Recursively build the dirs if they do not already exist
        os.makedirs(newspaper_list_file, exist_ok=True)
    except OSError:
        print('FAILED TO CREATE DIR RECURSIVELY')

    with open(newspaper_list_file + '/' + COMPANY_PATH, 'w') as filehandle:
        for listitem in newspaper_list:
            # print(listitem)
            filehandle.write('%s\n' % listitem)

def create_word_corpus(articles, company_tag):
    word_corpus = {}
    unique_words = []
    print('STARTING TO CREATE WORD CORPUS...')

    for article in articles.data:
        # Extract the parts of the article
        article_parts = article.split('\r\n')
        
        topic_name = article_parts[0].strip()

        try:
            author_name = article_parts[1].strip()[1:]
        except IndexError:
            print('Failed to return a author on article ' + article )

        try:
            article_date = article_parts[2].strip()
        except IndexError:
            print('Failed to return a valid date on article ' + article )
        

        # print('Topic name = ' + topic_name)
        # print('Author name = ' + author_name)
        # print('Article date = ' + article_date)
        # print('Link retrieved article from = ' + article_link)

        # Loop over every paragraph in the article
        for part in article_parts[4:]:
            # Catches the full sentences (used for sentiment analysis)
            sentences = part.split('.')

            # Loop through each sentence in paragraph
            for sentence in sentences:
                words = sentence.split(' ')
                # Loop through each word in the sentence
                for word in words:
                    # Ensure a word and not a number
                    if word.isalpha():
                        if word not in unique_words:
                            unique_words.append(word)
                            # New element so add it to the dictionary only after instantiating
                            word_corpus[word] = []
                            word_corpus[word].append(topic_name + ':::' + author_name + ':::' + article_date + ':::' + sentence)
                        else:
                            word_corpus[word].append(topic_name + ':::' + author_name + ':::' + article_date + ':::' + sentence)

    # for word in word_corpus:
    #     print('WORD: ' + word + ' POINTS TO ', word_corpus[word])

    # Add the bigrams to the word corpus
    print('Starting to add bigrams to the word corpus...')
    bigram_word_corpus = retrieve_bigram_word_corpus(company_tag)
    topic_name_bigram = ''
    author_name_bigram = ''
    article_date_bigram = ''
    sentence_bigram = ''

    # Convert bigram_word_corpus to a list:
    for bigram_word in bigram_word_corpus:
        bigram_word_list = []
        topic1_name = topic2_name = author1_name = author2_name = article1_date = article2_date = article1_sentence = article2_sentence = ''

        # Get words frm bigram
        print(bigram_word)

        
        if bigram_word.count('_') == 1:
            # Get bigrams
            word1, word2 = bigram_word.split('_')
        elif bigram_word.count('_') == 2:
             # Get trigrams, but only include first and second word for now
             word1, word2, _ = bigram_word.split('_')
        elif bigram_word.count('_') == 3:
            # Get quadrams
            word1, word2, _, _ = bigram_word.split('_')
            # Get pentams
        elif bigram_word.count('_') == 4:
            word1, word2, _, _, _ = bigram_word.split('_')
        else:
            print('Should not reach here!!!')
            continue
           

        # print('word1 = ', word1)
        # print('word2 = ', word2)

        if word1 in word_corpus.keys():
            if word2 in  word_corpus.keys():
                # In future revisions don't include only last bigram in array, take maximum count
                for word1, word2 in zip(word_corpus[word1],  word_corpus[word2]):
                    # Get word_parts for word1
                    word_parts = word1.split(':::')
                    # Get topic_name from bigram
                    topic1_name = word_parts[0]
                    # Get author_name for bigram
                    author1_name = word_parts[1]
                    # Get article_date for bigram
                    article1_date = word_parts[2]
                    # Get article sentence for bigram
                    article1_sentence = word_parts[3]

                    # Grab the parts of the word2
                    word_parts = word2.split(':::')
                    # Get topic_name from bigram
                    topic2_name = word_parts[0]
                    # Get author_name for bigram
                    author2_name = word_parts[1]
                    # Get article_date for bigram
                    article2_date = word_parts[2]
                    # Get article sentence for bigram
                    article2_sentence = word_parts[3]
                
                    # For now have both if and else have same result, but the else will be more
                    # complicated in future revisions, taking account more data
                    if topic1_name == topic2_name:
                        topic_name_bigram = topic1_name
                    else:
                        topic_name_bigram = topic1_name
                    
                    if author1_name == author2_name:
                        author_name_bigram = author1_name
                    else:
                        author_name_bigram = author1_name

                    if article1_date == article2_date:
                        article_date_bigram = article1_date
                    else:
                        article_date_bigram = article1_date

                    if article1_sentence == article2_sentence:
                        sentence_bigram = article1_sentence
                    else:
                        sentence_bigram = article1_sentence

                    # Add the bigram
                    if bigram_word not in bigram_word_list:
                        bigram_word_list.append(bigram_word)
                        word_corpus[bigram_word] = []
                    else:
                        word_corpus[bigram_word].append(topic_name_bigram + ':::' + author_name_bigram + ':::' + article_date_bigram + ':::' + sentence_bigram)

    
    return word_corpus

def save_word_corpus(word_corpus, company_tag):
    WORD_CORPUS_PATH = "../newspaper_data/word_corpus/" + company_tag + '.txt'

    FULL_WORD_CORPUS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), WORD_CORPUS_PATH))

    with open(FULL_WORD_CORPUS_PATH, 'wb') as myFile:
        pickle.dump(word_corpus, myFile)

def retrieve_bigram_word_corpus(company_tag):
    WORD_CORPUS_PATH = "../newspaper_data/bigram_word_corpus/" + company_tag + '.txt'

    FULL_WORD_CORPUS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), WORD_CORPUS_PATH))

    with open(FULL_WORD_CORPUS_PATH, 'rb') as myFile:
         word_corpus = pickle.load(myFile)
    
    return word_corpus

def make_bigram_word_corpus(cleaned_data, company_tag):
    WORD_CORPUS_PATH = "../newspaper_data/bigram_word_corpus/" + company_tag + '.txt'
    FULL_WORD_CORPUS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), WORD_CORPUS_PATH))

    unique_bigram_list = []
    bigram_word_corpus = {}

    # Grab the bigrams so we can append them to our corpus
    for article in cleaned_data:
        for word in article:
            if '_' in word:
                if word not in unique_bigram_list:
                    bigram_word_corpus[word] = []
                    unique_bigram_list.append(word)
                    bigram_word_corpus[word].append('')
                else:
                    unique_bigram_list.append(word)
                    bigram_word_corpus[word].append('')
    
    for word in bigram_word_corpus:
        print(word, end=", ")
    
    with open(FULL_WORD_CORPUS_PATH, 'wb') as myFile:
        pickle.dump(bigram_word_corpus, myFile) 


def main():
    TEST_CORPUS_PATH = '../corpus/20news-bydate-test/'
    REAL_INDEPENDENT_PATH = '../corpus/irishArticles/INDEPENDENT'
    REAL_BBC_PATH = '../corpus/BBC/INDEPENDENT'
    # mallet_tag = 'mallet'
    # data_type_tag = 'CleanData'


    company_tags = ['INDEPENDENT', 'BBC', 'NEW-YORK-TIMES']
    test_tag = 'TEST'

    #### INDEPENDENT MODEL BEING BUILT ####

    # Grab articles from bunch
    independent_bunch = gen_bunch(REAL_INDEPENDENT_PATH, company_tags[0])
    print('Finished loading newspaper data...')


    # # Clean newsdata:
    independent_data_cleaned = clean_data(independent_bunch, company_tags[0])
    print('Finished cleaning data...')

    # save_newspaper_data for independent
    # save_data(independent_data_cleaned, company_tags[0])
    # print('Finished saving cleaned_data...')

    # Ran once to cretae bigram list in a file
    # Creates a list of all of the bigram words within the corpus and stores it in a fie
    # make_bigram_word_corpus(independent_data_cleaned, company_tags[0])

    # # Create word corpus
    # word_corpus = create_word_corpus(independent_bunch, company_tags[0])

    # save_word_corpus(word_corpus, company_tags[0])




    # # Algo for corpus that points to everythiung
    # # 1. Each word in dict points to a list with every sentence it is in
    # # 2. The list has topicname and date appended to start
    # # 3. Crux is looping through each word and assigning the sentence and all else
    # # 'abortion\r\n{ Lyndsey Telford \r\n


    # # # Create dictionary. This maps id to the word
    # word_dict = corpora.Dictionary(independent_data_cleaned)

    # # # Create corpus. This directly contains ids of the word and the frequency.
    # corpus = [word_dict.doc2bow(data) for data in independent_data_cleaned]
    # print('Finished creating corpus...')

    # Will map word id to word frequency
    # print(corpus[:1])

    # Print wname of word and freqnecy
    # for c in corpus[:1]:
    #     for word_id, frequency in c:
    #         print(word_dict[word_id], " = ", frequency)

    # # Create the lda model. When using the method below ut returns None
    # lda_model = build_lda_model(independent_data_cleaned, word_dict, corpus)
    # print('Finished building lda model...')

    # # Save the model (regular lda model)
    # save_model(lda_model, company_tags[0])
    # print('Finished saving lda model...')

    # Print out contents of lda_model
    # pprint(lda_model.print_topics())

    # lda_mallet_model = build_mallet_lda_model(independent_data_cleaned, word_dict, corpus)
    # print('Finished building mallet lda model...')

    # # Save the mallet model
    # save_model(lda_mallet_model, company_tags[0], mallet_tag)
    # print('Finished saving lda model...')

    # doc_lda = lda_mallet_model[corpus]

    # print(doc_lda)


    # ## Extract below visualisation information to visualize_lda.py ##

    # # Show Topics
    # pprint(lda_mallet_model.show_topics(formatted=False))

    # # Compute Coherence Score
    # print('Starting to build coherence score...')
    # coherence_model_ldamallet = CoherenceModel(model=lda_mallet_model, texts=independent_data_cleaned, dictionary=word_dict, coherence='c_v')
    # coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    # print('\nCoherence Score: ', coherence_ldamallet)





if __name__ == '__main__':
    main()
