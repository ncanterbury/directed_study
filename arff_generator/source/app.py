import arff_control
import attributes
import configure
import manage_text
import tweet


class AppHandler:
    def __init__(self, config):
        self.config = config
        self.neutral_tweets = configure.get_neutral_tweets(config)
        self.data_frame = None
        self.is_labelled_data = None 
        self.election_data = None 

    def initialize_tweets(self):
        ''' we are creating the csv version of the tweets
            and then removing all neutral tweets
        '''
        print ("Beginning to initialize tweets")
        if configure.get_is_labelled_data(self.config) == "False":
            self.is_labelled_data = False 
        else:
            self.is_labelled_data = True
        
        tweet_path = configure.get_tweet_path(self.config)
        is_labelled_data = self.is_labelled_data
        tweet_handler = tweet.TweetHandler(tweet_path, is_labelled_data)

        # Build csv version of original text file
        tweet_handler.create_csv()

        '''
        No longer need to remove neutral
        '''
        #tweet_handler.remove_neutral()
        self.neutral_tweets = tweet_handler.tweets_csv

    def initialize_text(self):
        '''
        Initialize text will preprocess the neutral tweets. Includes functions
        for tokenizing, tokenize porter and removing stop words

        '''
        print ("Beginning to initialize text")
        is_labelled_data = self.is_labelled_data
        neutral_tweets = self.neutral_tweets
        text_handler = manage_text.TextHandler(neutral_tweets, is_labelled_data)
        text_handler.initialize_df()

        # save the dataframe build in the TextHandler to the app class
        if self.is_labelled_data:
            self.data_frame = text_handler.df
            self.election_data = None 
        else:
            self.data_frame = None 
            self.election_data = text_handler.election_data

    def initialize_attributes(self):
        print ("Beginning to initialize attributes")
        is_labelled_data = self.is_labelled_data

        attribute_handler = attributes.AttributeHandler(self.config, self.data_frame, self.election_data, is_labelled_data)
        attribute_handler.initialize_attributes()
        attribute_handler.initialize_data(self.neutral_tweets)
        attribute_handler.run_using_attributes()

        return attribute_handler

    def initialize_arff(self, built_attributes):
        '''
        When we initialize the arff, the data values should be passed to it
        from the built array of data in the attributes class.
        '''
        print("Beginning to initialize arff")
        attributes = built_attributes.attributes

        data = built_attributes.arff_data

        #neutral_tweets = self.neutral_tweets
        arff_path = configure.get_arff_path(self.config)
        is_labelled_data = self.is_labelled_data
        arff_handler = arff_control.ArffHandler(arff_path, attributes, data, is_labelled_data)

        arff_handler.generate_arff()
	print("Done")


def main():
    config_file = configure.args()
    if config_file == False:
        raise ValueError('Invalid args supplied')
    app = AppHandler(config_file)
    app.initialize_tweets()
    app.initialize_text()
    built_attributes = app.initialize_attributes()

    app.initialize_arff(built_attributes)

if __name__ == "__main__":
    main()
