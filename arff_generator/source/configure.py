import argparse
import os
import yaml


def get_neutral_tweets(config_file):
    with open(config_file, 'r') as stream:
        data=yaml.load(stream)
        return data['neutral_tweet_file']

def get_arff_path(config_file):
    with open(config_file, 'r') as stream:
        data=yaml.load(stream)
        return data['arff_file']

def get_is_labelled_data(config_file):
    with open(config_file, 'r') as stream:
        data=yaml.load(stream)
        return data['is_labelled_data'].title()

def handle_arff(config_file, path):
    fname = config_file
    with open(fname, 'r') as stream:
        data = yaml.load(stream)
        data['arff_file'] = path

    with open(fname, 'w') as yaml_file:
        yaml_file.write(yaml.dump(data, default_flow_style=False))


def handle_is_labelled(config_file, is_labelled_arg):
    fname = config_file
    with open(fname, 'r') as stream:
        data = yaml.load(stream)
        data['is_labelled_data'] = is_labelled_arg

    with open(fname, 'w') as yaml_file:
        yaml_file.write(yaml.dump(data, default_flow_style=False))

def get_tweet_path(config_file):
    with open(config_file, 'r') as stream:
        data = yaml.load(stream)
    return data['tweet_file']

def valid_path(config_file, path, arg_name):
    if not os.path.isfile(path):
        return False
    else:
        fname = config_file
        with open(fname, 'r') as stream:
            data = yaml.load(stream)
            data[arg_name] = path

        with open(fname, 'w') as yaml_file:
            yaml_file.write(yaml.dump(data, default_flow_style=False))
        return True

def args():
    '''
    Read args (path to tweets)
    '''
    parser = argparse.ArgumentParser(description ='Read in name of text file')
    parser.add_argument('tweets', help="system location of tweet file")
    parser.add_argument('arff_path', help="system location of tweet file")
    parser.add_argument('is_labelled_data', help="is this training data, or unlabelled data")

    args = parser.parse_args()

    # set config file based on if labelled data
    if args.is_labelled_data.lower() == "true":
        config_file = "source/labelled_data_config.yaml"
    else:
        config_file = "source/election_data_config.yaml"
        
    # set arff file in config
    handle_arff(config_file, args.arff_path)
    handle_is_labelled(config_file, args.is_labelled_data)
    # Check validity of tweet doc path
    if not valid_path(config_file, args.tweets, 'tweet_file'):
        print("Invalid file path for tweets")
        return False
    else:
        return config_file
