import arff
import yaml
from pprint import pprint

class ArffHandler:
    def __init__(self, arff_path, attributes, data, is_labelled_data):
        self.arff_path = arff_path
        self.attributes = attributes
        self.data = data
        self.is_labelled_data = is_labelled_data

    def initialize_attributes(self):
        with open('source/config.yaml', 'r') as f:
            doc = yaml.load(f)

        for att_val in doc['attributes']:
            if doc['attributes'][att_val]['multiple']:
                if self.is_labelled_data:
                    self.attributes.append((att_val, doc['attributes'][att_val]['value']))
                else:
                    self.attributes.append((att_val, '?'))
            else:
                self.attributes.append((att_val, doc['attributes'][att_val]['value'][0]))


    def generate_arff(self):
        # we want line count to figure out how many row in data
        # we will need
        attributes = self.attributes
        data = self.data

        obj = {
            'description': u'Twitter Data',
            'relation': 'tweets',
            'attributes': attributes,
            'data': data,
        }
        #pprint(obj)
        f = open(self.arff_path, 'wb')
        arff.dump(obj, f)
        f.close()


def main():
    handle = ArffHandler('arff_files/labelled_tweets.arff')
    #handle.generate_arff()
    handle.initialize_attributes()
    return

if __name__ == "__main__":
    main()
