import csv
import pandas as pd
import preprocessor as p
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

from nltk.stem import *
from nltk.corpus import stopwords
from nltk.stem.porter import *

class TextHandler():
    def __init__(self, neutral_tweets, is_labelled_data):
        self.neutral_tweets = neutral_tweets
        self.df = None
        self.stop = stopwords.words('english')
        self.is_labelled_data = is_labelled_data
        self.election_data = None 

    def initialize_df(self):
        print ("Running initialize dataframe")
        df = pd.DataFrame()

        if (self.is_labelled_data):

            train = pd.read_csv(self.neutral_tweets, header=0, encoding='ascii', delimiter=",", quoting=3, engine='c')
            print ("Loaded into dataframe")
            for col in range(0, len(train["tweet"])):
                if train['obama_score'][col] == 'obama=+1' or train['romney_score'][col] == 'romney=-1':
                    sentiment='pro_obama'
                elif train['romney_score'][col] == 'romney=+1' or train['obama_score'][col] == 'obama=-1':
                    sentiment='pro_romney'
                else:
                    sentiment='neutral'
                df = df.append([[train['tweet'][col], sentiment]], ignore_index=True)

            df.columns = ['tweet', 'sentiment']
            df["tweet"] = df["tweet"].apply(self.preprocessor)
            self.df = df
            print ('Loaded into dataframe')
            print df

        else:
            with open('/root/directedstudyfall17/arff_generator/tweets/preprocessed_election_data.csv', 'w') as outfile:
                # This second path should be changed to self.neutral tweets 
                with open('tweets/second_election_data_tweets.csv', 'r') as in_file:
                    reader = csv.reader(in_file)
                    line_number = 1
                    sentiment = "?"
                    for row in reader:
                        if ((line_number % 500000) == 0):
                            print line_number
                        line_number += 1
                        if line_number == 1000:
                            break
                        tweet = row[0]
                        
                        preprocessed_tweet = self.preprocessor(tweet)
                        #if ((line_number > 0 and line_number < 200000)):
                        if (re.search('[a-zA-Z]', preprocessed_tweet)):
                            preprocessed_tweet += ",?\n"
                            outfile.write(preprocessed_tweet)
            self.election_data = 'tweets/preprocessed_election_data.csv'

        print("Finished Initialize Dataframe")

    def preprocessor(self, text):
        text = ' '.join([w for w in text.split() if not w.startswith("http")])
        text = text.replace("RT ", "")
        text = text.replace("'", "")
        text = text.replace(",", "")
        punctuation = re.findall('(?:!|\?)', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower()) + \
               ' '.join(emoticons).replace('-', '') + \
               ' '.join(punctuation)
        return text

    def tokenizer(self, text):
        return text.split()

    def tokenizer_porter(self, text):
        porter = PorterStemmer()
        return [porter.stem(word) for word in text.split()]

    def stopwords(self, text):
        return ' '.join([w for w in self.tokenizer_porter(text) if w not in self.stop])

def main():
    handle = TextHandler('tweets/labelledtweets.csv', False)
    handle.initialize_df()
    data_frame = handle.df
    data_frame['tweet'] = data_frame['tweet'].apply(handle.preprocessor)

    #for tweet in data_frame['tweet']:
    #    print tweet
    #return

if __name__ == "__main__":
    main()
