import nltk
import csv
import numpy as np
import yaml
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

import manage_text
from ark_tweet.ark_tweet_nlp_python import CMUTweetTagger as cmu_tagger

count = CountVectorizer()
nltk.download('averaged_perceptron_tagger')


class AttributeHandler:
    def __init__(self, config_path, data_frame, election_tweets, is_labelled_data):
        self.config_path = config_path
        self.attributes = []
        self.function_names = {}
        self.arff_data = []
        self.available_attributes = {'bagofwords': self.bag_of_words, 'pos_tagging': self.pos_tagger, 'positive_words':
            self.positive_words, 'negative_words': self.negative_words, 
            'sentiment': self.sentiment,
            'vader_polarity': self.vader_polarity, 'sentence_length': self.sentence_length, 'text': self.only_text,
            'cmu_pos_tagging': self.cmu_pos_tagger, 
            'mention_romney': self.mention_romney, 
            'mention_obama': self.mention_obama}
        self.using_attributes = {}
        self.data_frame = data_frame
        self.neutral_tweets = None
        self.election_tweets = election_tweets
        self.is_labelled_data = is_labelled_data

    def initialize_attributes(self):
        ''' Read in the selected attributes from the config,
            and initialize them in the self.using_attributes array,
            with their values as the functions to be called
        '''
        with open(self.config_path, 'r') as f:
            doc = yaml.load(f)

        for att_val in doc['attributes']:
            self.using_attributes[att_val] = self.available_attributes[att_val]

    def add_attribute_name(self, att_val):
        ''' This function is individually called by each function as it is
            run. Allows for more control over how attribute names are added to
            self.attributes. This was necessary to deal with attributes such as pos tagging
            where we want one pos_tagging function, but want an attribute for every possible
            pos tag.
        '''
        with open(self.config_path, 'r') as f:
            doc = yaml.load(f)

        attribute = doc['attributes'][att_val]
        if attribute['multiple']:
            self.attributes.append((att_val, attribute['value']))
        else:
            self.attributes.append((att_val, attribute['value'][0]))


    def initialize_data(self, neutral_tweets):
        ''' Create data array of len(neutral_tweets) with all 0's
        '''
        print ("Attributes: Initializing Data")
        count = 0
        if self.is_labelled_data:

            with open(neutral_tweets, 'r') as data:
                reader = csv.reader(data)
                for line in reader:
                    count+=1
            print ("Arff Data Array Count", count)
            blank_data = [[] for x in range(count-1)]
        else:
            with open(self.election_tweets, 'r') as data:
                reader = csv.reader(data)
                for line in reader:
                    count+=1
            print ("Arff Data Array Count", count)
            blank_data = [[] for x in range(count)]


        self.arff_data = blank_data
        self.neutral_tweets = neutral_tweets

    def run_using_attributes(self):

        with open(self.config_path, 'r') as f:
            doc = yaml.load(f)

        for attr in doc['attributes']:
            if attr in self.using_attributes:
                self.using_attributes[attr](attr)

    def only_text(self, attribute_name):
        ''' stores the preprocessed text as an attribute
            done so we can later use WordToVector class in Weka
        '''
        self.add_attribute_name(attribute_name)
        arff_data_count = 0

        modify_text = manage_text.TextHandler(self.neutral_tweets)
        only_text_data_frame = self.data_frame.copy(deep=True)
        only_text_data_frame['tweet'] = only_text_data_frame['tweet'].apply(modify_text.stopwords)


        for tweet in only_text_data_frame['tweet']:
            self.arff_data[arff_data_count].append(tweet)
            arff_data_count+=1

    def bag_of_words(self, attribute_name):
        '''
        Will append attributes that it builds to self.arff_data
        '''
        self.add_attribute_name(attribute_name)

        #Want to experiment with stemming and removing stop words
        # specifically for bag of words generation



        modify_text = manage_text.TextHandler(self.neutral_tweets)
        bag_of_words_data_frame = self.data_frame.copy(deep=True)
        bag_of_words_data_frame['tweet'] = bag_of_words_data_frame['tweet'].apply(modify_text.stopwords)


        # we need to create a single attribute for every possible word
        # in the bag of words array

        np.set_printoptions(threshold=np.nan)

        count = CountVectorizer()
        string_array = []

        for tweet in bag_of_words_data_frame['tweet']:
            string_array.append(tweet)
        docs = np.array(string_array)
        #print docs
        bag = count.fit_transform(docs)
        unique_array = bag.toarray()


        for num in range(len(unique_array)):
            self.arff_data[num].append(sum(unique_array[num]))

    def positive_words(self, attribute_name):
        '''Build array with positive words, then loop over all tweets,
            check words for membership in list, and build count
        '''
        self.add_attribute_name(attribute_name)
        pos_per_tweet = []

        pos_list = []
        with open('pos_neg_words/positive-words.txt', 'r') as pos:
            for line in pos:
                line = line.replace("\n", "")
                pos_list.append(line)

        if self.is_labelled_data:
            for tweet in self.data_frame['tweet']:
                tokenize_tweet = word_tokenize(tweet)
                pos_count = 0
                for token in tokenize_tweet:
                    if token in pos_list:
                        pos_count+=1

                pos_per_tweet.append(pos_count)

            for num in range(len(pos_per_tweet)):
                self.arff_data[num].append(pos_per_tweet[num])
        else:
            with open(self.election_tweets, "r") as f:
                reader = csv.reader(f)
                row_count = 0
                for row in reader:
                    tokenize_tweet = word_tokenize(row[0])
                    pos_count = 0
                    for token in tokenize_tweet:
                        if token in pos_list:
                            pos_count+=1

                    self.arff_data[row_count].append(pos_count)
                    row_count += 1

    def negative_words(self, attribute_name):
        self.add_attribute_name(attribute_name)

        print("Attributes: Running Negative Words")

        neg_per_tweet = []

        neg_list=[]
        with open('pos_neg_words/negative-words.txt', 'r') as neg:
            for line in neg:
                line = line.replace("\n", "")
                neg_list.append(line)

        if self.is_labelled_data:
            for tweet in self.data_frame['tweet']:
                tokenize_tweet = word_tokenize(tweet)
                neg_count = 0
                for token in tokenize_tweet:
                    if token in neg_list:
                        neg_count+=1
                neg_per_tweet.append(neg_count)

            for num in range(len(neg_per_tweet)):
                self.arff_data[num].append(neg_per_tweet[num])
        else:
            with open(self.election_tweets, 'r') as f:
                reader = csv.reader(f)
                row_count = 0
                for row in reader:
                    if row_count % 500000 == 0:
                        print row_count
                    tokenize_tweet = word_tokenize(row[0])
                    neg_count = 0
                    for token in tokenize_tweet:
                        if token in neg_list:
                            neg_count+=1

                    self.arff_data[row_count].append(neg_count)
                    row_count += 1

    def vader_polarity(self, attribute_name):
        self.add_attribute_name(attribute_name)
        vader = SentimentIntensityAnalyzer()

        print ("Attributes: Starting Vader Polarity")

        # keep track of which data array we are adding to
        data_array_count = 0


        if self.is_labelled_data:
            for tweet in self.data_frame['tweet']:
                score = vader.polarity_scores(tweet)
                val = 0
                if score['neu'] > score['pos']:
                    val = 0
                else:
                    val = 1

                self.arff_data[data_array_count].append(val)
                data_array_count+=1
        else:
            with open(self.election_tweets, 'r') as f:
                reader = csv.reader(f)
                print len(self.arff_data) 
                row_count = 0
                for row in reader:
                    # row[0] is tweet 
                    score = vader.polarity_scores(row[0])
                    val = 0
                    if score['neu'] > score['pos']:
                        val = 0
                    else:
                        val = 1
                    if data_array_count % 500000 == 0:
                        print data_array_count
                    self.arff_data[data_array_count].append(val)
                    data_array_count += 1

    def sentence_length(self, attribute_name):
        self.add_attribute_name(attribute_name)
        print ("Attributes: Starting sentence length")

        # keep track of which data array we are adding to
        data_array_count = 0
        if self.is_labelled_data:
            for tweet in self.data_frame['tweet']:
                sentence_len = len(tweet)
                self.arff_data[data_array_count].append(sentence_len)
                data_array_count += 1
        else:
            with open(self.election_tweets, 'r') as f:
                reader = csv.reader(f)
                line_count = 0
                for row in reader:
                    sentence_len = len(row[0])
                    line_count += 1
                    if line_count % 500000 == 0:
                        print line_count 
                    self.arff_data[data_array_count].append(sentence_len)
                    data_array_count += 1
           
    def mention_obama(self, attribute_name):
        self.add_attribute_name(attribute_name)

        # keep track of which data array we are adding to
        data_array_count = 0

        if self.is_labelled_data:
            for tweet in self.data_frame['tweet']:
                if 'obama' in tweet.split():
                    self.arff_data[data_array_count].append(1)
                else:
                    self.arff_data[data_array_count].append(0)
                 
                data_array_count += 1
        else:
            with open(self.election_tweets, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    tweet = row[0]
                    if 'obama' in tweet.split():
                        self.arff_data[data_array_count].append(1)
                    else:
                        self.arff_data[data_array_count].append(0)
                     
                    data_array_count += 1


    def mention_romney(self, attribute_name):
        self.add_attribute_name(attribute_name)
        print("Atrributes: Starting Mention Romney")

        # keep track of which data array we are adding to
        data_array_count = 0

        if self.is_labelled_data:
            for tweet in self.data_frame['tweet']:
                if 'romney' in tweet.split():
                    self.arff_data[data_array_count].append(1)
                else:
                    self.arff_data[data_array_count].append(0)        
                data_array_count += 1
        else:
            with open(self.election_tweets, 'r') as f:
                reader = csv.reader(f)
                line_count = 0
                for row in reader:
                    if line_count % 500000 == 0:
                        print line_count
                    line_count += 1
                    tweet = row[0]
                    if 'romney' in tweet.split():
                        self.arff_data[data_array_count].append(1)
                    else:
                        self.arff_data[data_array_count].append(0)
                     
                    data_array_count += 1

                        
    def cmu_pos_tagger(self, attribute_name):
        '''
        like_pos tagger, cmu tagger will have to add each one of the pos
        labels to the attributes list, along with the attribute value after generation.
        diff from how other functions add name to attributes
        '''

        print ("Attributes: Beginning CMU POS Tagger")

        tagset_dict = {}
        if self.is_labelled_data:
            tagset_options_list = ['O', 'V', 'D', 'A', 'N', ',', '^', 'L', '~', '@', 'U', '$', 'E', '!',
                               '&', 'R', '#', 'G', 'T', 'M', 'X', 'S', 'Z', 'Y']
        else:
            tagset_options_list = ['T']
        tweet_list = []
        # add tags to attribute name
        for tag in tagset_options_list:
            self.attributes.append((tag, 'NUMERIC'))
            tagset_dict[tag] = 0

        # keep track of which data array we are adding to
        data_array_count = 0

        # make a copy of the data array in case we want to stem/remove
        # stop words without altering original array
        if self.is_labelled_data:
            cmu_tagger_data_frame = self.data_frame.copy(deep=True)
        else:
            cmu_tagger_data_list = []
            with open(self.election_tweets, 'r') as f:
                reader = csv.reader(f)
                print ("Creating cmu tagger data frame")

                line_number = 0
                for row in reader:
                    if line_number % 500000 == 0:
                        print line_number
                    line_number += 1
                    cmu_tagger_data_list.append(row[0])

        # build list of all tweets before pos tagging
        if self.is_labelled_data:
            stored_tweets = cmu_tagger_data_frame['tweet']
        else:
            stored_tweets = cmu_tagger_data_list
        print ("Length of stored tweets:", len(stored_tweets))
        
        for tweet in stored_tweets: 
            if (tweet == ""):
                tweet_list.append("Link")
            else:
                tweet_list.append(str(tweet))

        print("Attributes: CMU Tagger: Successfully built tweet list. Running tagger")

        tagged_tweets = cmu_tagger.runtagger_parse(tweet_list)
        print ("Finished Tagger")
        print ("Writing Tagged Tweets to Arff Data")
        print ("")
        print ("tweet count")
        print len(tagged_tweets)
        tweet_count = 0
        for tweet in tagged_tweets:
            if tweet_count % 500000 == 0:
                print tweet_count
            tweet_count += 1
            for tagged_word in tweet:
                if tagged_word[1] in tagset_dict:
                    if tagged_word[1] != 'P':
                        # modify count of tag_dict for tweet
                        tagset_dict[tagged_word[1]] += 1

            for tags in tagset_options_list:
                self.arff_data[data_array_count].append(tagset_dict[tags])

            data_array_count+=1

            # reset tagset dict values to 0 for next word
            for tag in tagset_dict:
                tagset_dict[tag] = 0

    def pos_tagger(self, attribute_name):
        '''
        pos tagger will have to add each one of the pos labels to the
        attributes list, along with the attribute value after generation.
        diff from how other functions add name to attributes
        '''
        tagset_dict = {}
        tagset_options_list=['.', ':', ')', '$', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP',
                        'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',
                        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        # Add tags to attribute names
        for tag in tagset_options_list:
            self.attributes.append((tag, 'NUMERIC'))
            tagset_dict[tag] = 0

        #stem and remove stop words before pos tagging
        modify_text = manage_text.TextHandler(self.neutral_tweets, True)
        pos_tagger_data_frame = self.data_frame.copy(deep=True)
        pos_tagger_data_frame['tweet'] = pos_tagger_data_frame['tweet'].apply(modify_text.stopwords)

        # keep track of which data array we are adding to
        data_array_count=0

        # tag each tweet
        for tweet in pos_tagger_data_frame['tweet']:

            token_tweet = word_tokenize(tweet)
            tagged_tweet = nltk.pos_tag(token_tweet)

            # modify count of tag dict for tweet
            for word in tagged_tweet:
                tagset_dict[word[1]]+=1

            for tags in tagset_options_list:
                self.arff_data[data_array_count].append(tagset_dict[tags])
            data_array_count+=1

            # Reset tagset_dict values to 0 for next word
            for tag in tagset_dict:
                tagset_dict[tag] = 0
        return


    def sentiment(self, attribute_name):
        print("Attributes: Running Sentiment")
        self.add_attribute_name(attribute_name)
        sentiment_array = []

        if self.is_labelled_data:
            for sent in self.data_frame['sentiment']:
                sentiment_array.append(sent)

            for num in range(len(sentiment_array)):
                self.arff_data[num].append(sentiment_array[num])
        else:
            with open(self.election_tweets, 'r') as f:
                reader = csv.reader(f)
                row_num = 0
                for row in reader:
                    if row_num % 500000 == 0:
                        print row_num
                    self.arff_data[row_num].append("?")
                    row_num+=1


def main():

    # Initialize text handler
    #text_handler = manage_text.TextHandler('tweets/neutral_labelledtweets.csv', False)
    #text_handler.initialize_df()
    #data_frame = text_handler.df
    #data_frame['tweet'] = data_frame['tweet'].apply(text_handler.preprocessor)
    data_frame = None 
    #election_data = text_handler.election_data
    election_data = "/root/directedstudyfall17/arff_generator/tweets/preprocessed_election_data.csv"
    handle = AttributeHandler('source/config.yaml', data_frame, election_data, False)
    handle.initialize_attributes()
    handle.initialize_data('tweets/preprocessed_election_data.csv')
    #handle.vader_polarity('hello')
    #handle.negative_words('hello')
    #handle.sentence_length('hello')
    #handle.mention_romney('hello') 
    handle.cmu_pos_tagger('hello')
    #print data_frame

    '''
    #handle.generate_arff()
    handle.initialize_attributes()
    handle.initialize_data('tweets/neutral_labelledtweets.csv')
    handle.run_using_attributes(data_frame)
    '''
    return

if __name__ == "__main__":
    main()
