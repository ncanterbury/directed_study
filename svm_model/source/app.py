import configure 
import yaml
import pandas as pd 
from svm_model import SVM_Model


def initialize_svm():
    fname = 'source/config.yaml'
    with open(fname, 'r') as stream:
        yaml_file = yaml.load(stream)
        #feature_selection = yaml_file['feature_selection']
        grid_search = yaml_file['grid_search']
        #svm_cost = yaml_file['svm_cost']
        #svm_gamma = yaml_file['svm_gamma']
        csv_file = yaml_file['csv_file']

    data_frame = pd.read_csv(csv_file) 
    if grid_search.lower() == "false":
        grid_search = False 
    else:
        grid_search = True 
    SVM = SVM_Model(data_frame, grid_search)
    
    return SVM

def run_svm(SVM):
    SVM.make_test_train()
    #SVM.standardize_data()
    SVM.univariate_feature_selection()
    #SVM.recursive_feature_selection()
    #SVM.select_from_model()
    SVM.svm_model()
    if SVM.grid_search == True:
        SVM.run_grid_search()
    #SVM.roc_area()


def main():
    # run args parser, initialize config 
    configure.args()
    SVM = initialize_svm()
    run_svm(SVM)

if __name__ == "__main__":
    main()
