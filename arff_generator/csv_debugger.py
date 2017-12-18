import csv


def debug():
    with open(r"/root/directedstudyfall17/arff_generator/tweets/election_data_tweets.csv", 'r') as f:
        reader = csv.reader(f)
        linenumber = 1
        try:
            for row in reader:
                if linenumber % 10000 == 0:
                    print linenumber
                print row 

                linenumber += 1
        except Exception as e:
            print (("Error line %d: %s %s" % (linenumber, str(type(e)), e.message)))
def main():
    debug()

if __name__ == "__main__":
    main()
