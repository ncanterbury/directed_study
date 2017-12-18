import csv
import yaml

class TweetHandler:
    def __init__(self, tweet_path, is_labelled_data):
        self.tweet_path = tweet_path
        self.tweets_csv = None
        self.tweets_neutral = None
        self.is_labelled_data = is_labelled_data
    
    def create_csv(self):
        if self.is_labelled_data == True:
            self.create_labelled_csv()
        else:
            self.create_unlabelled_csv()

    def create_labelled_csv(self):
        # change this to not be hardcoded in function
        csv_path = 'tweets/labelledtweets.csv'
        with open(self.tweet_path, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            line_list = [line for line in stripped]

            with open(csv_path, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(('tweet', 'obama_score', 'romney_score'))
                for count in range(0, len(line_list), 2):
                    tweet = line_list[count].replace(',', ' ')
                    scores = line_list[count + 1].split(",")
                    obama_score = scores[0]
                    romney_score = scores[1]
                    writer.writerow((tweet, obama_score, romney_score))
            self.tweets_csv = csv_path


    def create_unlabelled_csv(self):
        print ("Tweets: Create Unlabelled CSV")
        # change this to not be hardcoded in function
        csv_path = 'tweets/second_election_data_tweets.csv'
	'''
        with open('tweets/election_data_tweets.txt', 'w') as f2:
            with open(self.tweet_path) as f:
                for line in f:
                    words = line.split(";@;")
                    f2.write(words[-1])
        print("Made it past the first one")
	'''
        with open(csv_path, 'w') as out_file:
            writer = csv.writer(out_file)
            # dont think we need to write labels. Only needed that when the data
            # was read into a pandas df 
            #writer.writerow(('tweet', 'sentiment'))
            scores = "?"
            row_count = 0
            with open(self.tweet_path) as in_file:
                for line in in_file:
                    if row_count % 500000 == 0:
                        print row_count 
                    row_count += 1
                    words = line.split(";@;")
                    line = words[-1]
                    strip_line = line.strip()
                    tweet = strip_line.replace(',', ' ')
                    writer.writerow((tweet, scores))
                    #out_file.write(words[-1])
            
            self.tweets_csv = csv_path

        print("Tweets: Finished Create Unlabelled CSV")
        print("")

        '''
        with open(csv_path, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('tweet', 'sentiment'))
            scores = "?"
            count = 0
            with open('tweets/election_data_tweets.txt') as in_file:
                for line in in_file:
                    if count == 0:
                        count += 1
                        pass
                    else:
                        strip_line = line.strip()
                        tweet = strip_line.replace(',', ' ')
                        writer.writerow((tweet, scores))
            self.tweets_csv = csv_path 
        '''

        '''
        with open('tweets/election_data_tweets.txt', 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            line_list = [line for line in stripped]

            with open(csv_path, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(('tweet', 'sentiment'))
                for count in range(0, len(line_list)):
                    tweet = line_list[count].replace(',', ' ')
                    scores = "?" 
                    writer.writerow((tweet, scores))
            self.tweets_csv = csv_path
        '''

    def remove_neutral(self):
        neutral_tweet_path = 'tweets/neutral_labelledtweets.csv'
        with open(self.tweets_csv, 'r') as in_file:
            reader = csv.DictReader(in_file)

            with open(neutral_tweet_path, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(('tweet', 'obama_score', 'romney_score'))
                for row in reader:
                    if 'obama=na' in row['obama_score'] and 'romney=na' in row['romney_score']:
                        continue
                    elif 'obama=0' in row['obama_score'] and 'romney=na' in row['romney_score']:
                        continue
                    elif 'obama=na' in row['obama_score'] and 'romney=0' in row['romney_score']:
                        continue
                    elif 'obama=0' in row['obama_score'] and 'romney=0' in row['romney_score']:
                        continue
                    else:
                        writer.writerow((row['tweet'], row['obama_score'], row['romney_score']))

            with open("source/config.yaml", 'r') as stream:
                data = yaml.load(stream)
                data['neutral_tweet_file'] = neutral_tweet_path
            self.tweets_neutral = neutral_tweet_path

def main():
    tweet = TweetHandler("source/test_getting_tweet.txt", False)
    tweet.create_csv()

if __name__ == "__main__":
    main()
