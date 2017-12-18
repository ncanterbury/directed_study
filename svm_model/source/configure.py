import argparse
import yaml


def initialize_config(args):
    fname = "source/config.yaml"
    with open(fname, 'r') as stream:
        data = yaml.load(stream)
        #data['feature_selection'] = args.feature_selection
        data['grid_search'] = args.grid_search
        #data['svm_cost'] = args.svm_cost 
        #data['svm_gamma'] = args.svm_gamma
        data['csv_file'] = args.csv_file 

    with open(fname, 'w') as yaml_file:
        yaml_file.write(yaml.dump(data, default_flow_style=False))

def args():
    '''
    Read args (path to tweets)
    '''
    parser = argparse.ArgumentParser(description ='Read args for SVM model')
    #parser.add_argument('feature_selection', help="which feature selection should be run")
    parser.add_argument('grid_search', help="should grid search be run. If not, run SVM estimator with supplied parameters")
    #parser.add_argument('svm_cost', help="svm cost")
    #parser.add_argument('svm_gamma', help='svm gamma')
    parser.add_argument('csv_file', help='path to csv')

    args = parser.parse_args()
    initialize_config(args)
   
